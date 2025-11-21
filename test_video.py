import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model import EventDetector
import numpy as np
import torch.nn.functional as F

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


def display_key_frames_interactive(video_path, events, confidence):
    """
    Display key frames with interactive navigation.
    Use Left/Right arrow keys or A/D to navigate, Q or ESC to quit.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Pre-load all event frames
    event_frames = []
    for e in events:
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        ret, img = cap.read()
        if ret:
            event_frames.append(img.copy())
    cap.release()
    
    if not event_frames:
        print("No frames to display")
        return
    
    current_idx = 0
    window_name = 'Golf Swing Key Frames - Navigate with Arrow Keys (Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("\n" + "="*60)
    print("INTERACTIVE VIEWER CONTROLS:")
    print("  Right Arrow / D : Next frame")
    print("  Left Arrow / A  : Previous frame")
    print("  Q / ESC         : Quit")
    print("="*60 + "\n")
    
    while True:
        # Create a copy of the current frame to annotate
        display_img = event_frames[current_idx].copy()
        
        # Add event information
        event_name = event_names[current_idx]
        conf = confidence[current_idx]
        frame_num = events[current_idx]
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        thickness = 2
        
        # Event name (top)
        cv2.putText(display_img, f'{event_name}', 
                   (20, 40), font, font_scale, (0, 255, 0), thickness)
        
        # Confidence score
        cv2.putText(display_img, f'Confidence: {conf:.3f}', 
                   (20, 80), font, font_scale * 0.8, (0, 255, 255), thickness)
        
        # Frame number
        cv2.putText(display_img, f'Frame: {frame_num}', 
                   (20, 120), font, font_scale * 0.8, (255, 255, 255), thickness)
        
        # Navigation info (bottom)
        nav_text = f'[{current_idx + 1}/{len(event_frames)}] Use arrows/A/D to navigate, Q to quit'
        cv2.putText(display_img, nav_text, 
                   (20, display_img.shape[0] - 20), 
                   font, font_scale * 0.6, (200, 200, 200), 1)
        
        cv2.imshow(window_name, display_img)
        
        # Wait for key press (wait indefinitely until a key is pressed)
        key = cv2.waitKey(0)
        
        # Navigation controls - handle multiple key code possibilities
        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            break
        elif key == ord('d') or key == ord('D') or key == 2555904 or key == 83:  # D or Right arrow
            current_idx = min(current_idx + 1, len(event_frames) - 1)
            print(f"Moving to frame {current_idx + 1}/{len(event_frames)}: {event_names[current_idx]}")
        elif key == ord('a') or key == ord('A') or key == 2424832 or key == 81:  # A or Left arrow
            current_idx = max(current_idx - 1, 0)
            print(f"Moving to frame {current_idx + 1}/{len(event_frames)}: {event_names[current_idx]}")
        else:
            # Print key code to help debug if keys aren't working
            print(f"Key pressed: {key} (use A/D or arrow keys to navigate, Q to quit)")
    
    cv2.destroyAllWindows()
    print("\nViewer closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video that you want to test', default='test_video.mp4')
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=64)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    print('Testing...')
    for sample in dl:
        images = sample['images']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
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

    # Display frames interactively
    display_key_frames_interactive(args.path, events, confidence)