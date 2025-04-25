import os
import pandas as pd
from pathlib import Path
import random

def create_labels_csv(dataset_root):
    dataset_root = Path(dataset_root)
    data = []
    
    # Age ranges for different categories
    age_ranges = {
        'fractured': (12, 180),  # Fractured bones typically in younger patients
        'not fractured': (24, 240)  # Normal X-rays from wider age range
    }
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        split_dir = dataset_root / split
        
        # Process fractured images
        fractured_dir = split_dir / 'fractured'
        if fractured_dir.exists():
            for img in fractured_dir.glob('*.png'):
                age = random.randint(*age_ranges['fractured'])
                data.append({
                    'image': f"{split}/fractured/{img.name}",
                    'fracture': 1,
                    'age_months': age
                })
        
        # Process not fractured images
        not_fractured_dir = split_dir / 'not fractured'
        if not_fractured_dir.exists():
            for img in not_fractured_dir.glob('*.png'):
                age = random.randint(*age_ranges['not fractured'])
                data.append({
                    'image': f"{split}/not fractured/{img.name}",
                    'fracture': 0,
                    'age_months': age
                })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(dataset_root / 'labels.csv', index=False)
    print(f"Created labels.csv with {len(df)} entries")
    print(f"Age statistics:")
    print(f"Min age: {df.age_months.min()} months")
    print(f"Max age: {df.age_months.max()} months")
    print(f"Mean age: {df.age_months.mean():.1f} months")

if __name__ == '__main__':
    create_labels_csv('Bone_Fracture_Binary_Classification') 