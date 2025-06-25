import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SiameseFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []
        self.people = []

        self._create_pairs()

    def _create_pairs(self):
        genders = ['male', 'female']
        for gender in genders:
            gender_path = os.path.join(self.root_dir, gender)
            if not os.path.isdir(gender_path):
                continue
            for person_id in os.listdir(gender_path):
                person_path = os.path.join(gender_path, person_id)
                if not os.path.isdir(person_path):
                    continue
                clean_img = os.path.join(person_path, "clean.jpg")
                distortion_folder = os.path.join(person_path, "distortion")
                distorted_imgs = [os.path.join(distortion_folder, img) 
                                  for img in os.listdir(distortion_folder)
                                  if img.endswith(('.jpg', '.jpeg', '.png'))]
                self.people.append((clean_img, distorted_imgs))

                # Positive pairs: clean with each distorted
                for dist_img in distorted_imgs:
                    self.pairs.append((clean_img, dist_img, 1))

        # Create negative pairs (clean-clean)
        for _ in range(len(self.pairs)):
            (clean1, _), (clean2, _) = random.sample(self.people, 2)
            self.pairs.append((clean1, clean2, 0))

        # Additional negative pairs: distorted-distorted of different people
        for _ in range(len(self.pairs)):
            ( _, dist_list1), (_, dist_list2) = random.sample(self.people, 2)
            if dist_list1 and dist_list2:
                dist_img1 = random.choice(dist_list1)
                dist_img2 = random.choice(dist_list2)
                self.pairs.append((dist_img1, dist_img2, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.triplets = []
        self.label_map = {}

        # Gather all identities
        self.identities = [d for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))]

        for idx, identity in enumerate(self.identities):
            self.label_map[identity] = idx

        # Build triplets (anchor, positive, negative)
        for identity in self.identities:
            id_path = os.path.join(root_dir, identity)
            clean_img = None
            for f in os.listdir(id_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    clean_img = os.path.join(id_path, f)
                    break
            if not clean_img:
                continue

            distortion_path = os.path.join(id_path, "distortion")
            if not os.path.isdir(distortion_path):
                continue

            distorted_imgs = [
                os.path.join(distortion_path, f)
                for f in os.listdir(distortion_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            # Create triplets
            for pos_img in distorted_imgs:
                neg_identity = random.choice(
                    [i for i in self.identities if i != identity])
                neg_path = os.path.join(root_dir, neg_identity)
                neg_clean = None
                neg_distorted_imgs = []
                for f in os.listdir(neg_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        neg_clean = os.path.join(neg_path, f)
                    distortion_folder = os.path.join(neg_path, "distortion")
                    if os.path.isdir(distortion_folder):
                        neg_distorted_imgs = [os.path.join(distortion_folder, img)
                                             for img in os.listdir(distortion_folder)
                                             if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                neg_img_choice = neg_clean
                if neg_distorted_imgs:
                    neg_img_choice = random.choice(neg_distorted_imgs)
                if neg_img_choice:
                    self.triplets.append((clean_img, pos_img, neg_img_choice))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path = self.triplets[idx]
        anchor_img = Image.open(anchor_path).convert("RGB")
        pos_img = Image.open(pos_path).convert("RGB")
        neg_img = Image.open(neg_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img
