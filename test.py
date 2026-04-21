from ViT import ViT
import torch
import yaml
from utils import hardware_check, plot_prediction
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from tqdm import tqdm

# set device
device = hardware_check()

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load the pretrained model
# ViT inizialitazion
pretrained_model = ViT(
    in_channels=config["in_channels"],
    image_size=config["image_size"],
    patch_size=config["patch_size"],
    number_of_encoder=config["num_encoders"],
    embeddings=config["embedding_dim"],
    d_ff_scale=config["d_ff_scale"],
    heads=config["num_heads"],
    input_dropout_rate=config["input_dropout"],
    attention_dropout_rate=config["attention_dropout"],
    feed_forward_dropout_rate=config["ff_dropout"],
    number_of_classes=config["num_classes"]
).to(device)

pretrained_model.load_state_dict(torch.load('/path_to_the_model.pth'))

# Set the model to evaluation mode
pretrained_model.eval()

# Test Dataset
transform = transforms.Compose([
    transforms.Resize(config["resize"]),
    transforms.CenterCrop(config["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(config["normalize_mean"], config["normalize_std"])
])

test_dataset = datasets.ImageFolder(root="data_path"+"/afhq/val/", transform=transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# Test Loop
# set model to eval mode
pretrained_model.eval()
pretrained_model.to(device)

# initialize accumulators
test_loss = 0.0
test_acc = 0.0

loss_fn = torch.nn.CrossEntropyLoss()

all_preds = []
all_labels = []

# test loop
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        logits = pretrained_model(images)
        loss = loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        test_loss += loss.item()

test_acc = (np.array(all_preds) == np.array(all_labels)).mean()

print("\n==================== Test Results ==================")
print(f"Test Accuracy:    {test_acc:.4f}")
print("======================================================")


class2label = {0:"cat", 1:"dog", 2:"wild"}

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        logits = pretrained_model(images)
        preds = torch.argmax(logits, dim=1)

        plot_prediction(images[5], preds[5], labels[5], class2label)

        if i == 0:
          break