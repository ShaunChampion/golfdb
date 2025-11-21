import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model import EventDetector
import numpy as np
import torch.nn.functional as F
from pose_visualizer import PoseVisualizer

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}


class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # preprocess and return frames
        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images)) # only for compatibility with transforms
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def display_key_frames_with_pose(video_path, events, confidence, pose_visualizer):
    """
    Display key frames with pose overlay and interactive navigation.
    Use Left/Right arrow keys or A/D to navigate, Q or ESC to quit.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Pre-load all event frames and extract poses
    event_frames = []
    event_poses = []
    event_metrics = []
    
    print("\nExtracting poses for key frames...")
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        ret, img = cap.read()
        if ret:
            event_frames.append(img.copy())
            
            # Extract pose
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            keypoints = pose_visualizer.extract_pose(frame_rgb)
            event_poses.append(keypoints)
            
            # Calculate metrics
            metrics = pose_visualizer.get_golf_metrics(keypoints)
            event_metrics.append(metrics)
            
            print(f"  Frame {e} ({event_names[i]}): Pose extracted")
    
    cap.release()
    
    if not event_frames:
        print("No frames to display")
        return
    
    current_idx = 0
    window_name = 'Golf Swing Analysis - Navigate with Arrow Keys (Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("\n" + "="*60)
    print("INTERACTIVE VIEWER CONTROLS:")
    print("  Right Arrow / D : Next frame")
    print("  Left Arrow / A  : Previous frame")
    print("  S               : Save current frame")
    print("  P               : Toggle pose overlay")
    print("  Q / ESC         : Quit")
    print("="*60 + "\n")
    
    show_pose = True
    
    while True:
        # Get current frame data
        display_img = event_frames[current_idx].copy()
        keypoints = event_poses[current_idx]
        metrics = event_metrics[current_idx]
        
        event_name = event_names[current_idx]
        conf = confidence[current_idx]
        frame_num = events[current_idx]
        
        # Add pose overlay if enabled
        if show_pose:
            display_img = pose_visualizer.annotate_frame_with_metrics(
                display_img, keypoints, event_name, conf
            )
        else:
            # Just add basic text without pose
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(display_img, f'{event_name}', 
                       (20, 40), font, 0.8, (0, 255, 0), 2)
            cv2.putText(display_img, f'Confidence: {conf:.3f}', 
                       (20, 80), font, 0.6, (0, 255, 255), 2)
            cv2.putText(display_img, f'Frame: {frame_num}', 
                       (20, 120), font, 0.6, (255, 255, 255), 2)
        
        # Add navigation info
        nav_text = f'[{current_idx + 1}/{len(event_frames)}] Arrows=Navigate | S=Save | P=Toggle Pose | Q=Quit'
        cv2.putText(display_img, nav_text, 
                   (20, display_img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(window_name, display_img)
        
        # Print metrics to console
        print(f"\n--- {event_name} (Frame {frame_num}) ---")
        for metric_name, value in metrics.items():
            display_name = metric_name.replace('_', ' ').title()
            print(f"  {display_name}: {value:.1f}°")
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        # Navigation controls
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == 83 or key == ord('d'):  # Right arrow or D
            current_idx = min(current_idx + 1, len(event_frames) - 1)
        elif key == 81 or key == ord('a'):  # Left arrow or A
            current_idx = max(current_idx - 1, 0)
        elif key == ord('s'):  # Save frame
            filename = f'frame_{event_name.replace(" ", "_")}_{frame_num}.png'
            cv2.imwrite(filename, display_img)
            print(f"Saved frame to: {filename}")
        elif key == ord('p'):  # Toggle pose overlay
            show_pose = not show_pose
            print(f"Pose overlay: {'ON' if show_pose else 'OFF'}")
    
    cv2.destroyAllWindows()
    print("\nViewer closed.")


def save_analysis_report(video_path, events, confidence, pose_visualizer, output_path='analysis_report.txt'):
    """
    Save a text report of the swing analysis with metrics for each event.
    """
    cap = cv2.VideoCapture(video_path)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GOLF SWING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Total Frames Analyzed: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}\n")
        f.write(f"Key Events Detected: {len(events)}\n\n")
        
        for i, e in enumerate(events):
            cap.set(cv2.CAP_PROP_POS_FRAMES, e)
            ret, img = cap.read()
            
            if ret:
                # Extract pose and metrics
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                keypoints = pose_visualizer.extract_pose(frame_rgb)
                metrics = pose_visualizer.get_golf_metrics(keypoints)
                
                # Write to report
                f.write("-" * 80 + "\n")
                f.write(f"Event {i+1}: {event_names[i]}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Frame Number: {e}\n")
                f.write(f"Model Confidence: {confidence[i]:.3f}\n\n")
                
                if metrics:
                    f.write("Biomechanical Metrics:\n")
                    for metric_name, value in metrics.items():
                        display_name = metric_name.replace('_', ' ').title()
                        f.write(f"  • {display_name}: {value:.1f}°\n")
                else:
                    f.write("  [Insufficient pose confidence for metrics]\n")
                
                f.write("\n")
    
    cap.release()
    print(f"Analysis report saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video that you want to test', default='test_video.mp4')
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=64)
    parser.add_argument('--no-pose', action='store_true', help='Disable pose visualization')
    parser.add_argument('--save-report', action='store_true', help='Save analysis report to file')
    parser.add_argument('--save-video', type=str, help='Save annotated video to path')
    args = parser.parse_args()
    seq_length = args.seq_length

    print('Preparing video: {}'.format(args.path))

    ds = SampleVideo(args.path, transform=transforms.Compose([ToTensor(),
                                Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    try:
        save_dict = torch.load('models/swingnet_1800.pth.tar')
    except:
        print("Model weights not found. Download model weights and place in 'models' folder. See README for instructions")
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    print('Running event detection...')
    for sample in dl:
        images = sample['images']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.to(device))
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {}'.format(events))

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Confidence: {}'.format([np.round(c, 3) for c in confidence]))

    # Initialize pose visualizer
    if not args.no_pose:
        print("\nInitializing pose estimation...")
        pose_visualizer = PoseVisualizer(model_type='lightning')
        
        # Save analysis report if requested
        if args.save_report:
            save_analysis_report(args.path, events, confidence, pose_visualizer)
        
        # Save annotated video if requested
        if args.save_video:
            from pose_visualizer import process_video_with_pose
            print(f"\nGenerating annotated video...")
            process_video_with_pose(
                video_path=args.path,
                output_path=args.save_video,
                event_frames=list(events),
                event_names=[event_names[i] for i in range(len(events))],
                confidence_scores=confidence
            )
        
        # Display frames interactively with pose
        display_key_frames_with_pose(args.path, events, confidence, pose_visualizer)
    else:
        # Original display without pose
        cap = cv2.VideoCapture(args.path)
        for i, e in enumerate(events):
            cap.set(cv2.CAP_PROP_POS_FRAMES, e)
            _, img = cap.read()
            cv2.putText(img, '{:.3f}'.format(confidence[i]), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))
            cv2.imshow(event_names[i], img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()