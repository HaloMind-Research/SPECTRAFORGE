import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from dataset import DualStreamDataset
from models import SPECTRAFORGE
from tqdm import tqdm

def train_spectraforge(train_paths, train_labels, val_paths, val_labels, epochs=25, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(DualStreamDataset(train_paths, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DualStreamDataset(val_paths, val_labels), batch_size=batch_size, shuffle=False)
    
    model = SPECTRAFORGE().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    best_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for sp, fq, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            sp, fq, y = sp.to(device), fq.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sp, fq), y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for sp, fq, y in val_loader:
                out = torch.sigmoid(model(sp.to(device), fq.to(device)))
                all_preds.extend(out.cpu().numpy())
                all_targets.extend(y.numpy())
                
        auc = roc_auc_score(all_targets, all_preds)
        print(f"Epoch {epoch+1} | Val AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'spectraforge_best.pth')

if __name__ == "__main__":
    # Placeholder for path loading logic
    print("Initiating SPECTRAFORGE Training...")
