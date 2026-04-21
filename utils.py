import torch
import os
from rich.tree import Tree
from rich import print
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np  

def hardware_check():
    # Check if GPU is available and return the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available!")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")

    return device

def generate_directory_tree(percorso, tree=None, limite_file=4):
    nome = os.path.basename(percorso)
    if tree is None:
        tree = Tree(f"📂 [bold blue]{nome}[/]")

    elementi = sorted(os.listdir(percorso))
    cartelle = [e for e in elementi if os.path.isdir(os.path.join(percorso, e))]
    file = [e for e in elementi if os.path.isfile(os.path.join(percorso, e))]

    for directory in cartelle:
        ramo = tree.add(f"📁 [bold magenta]{directory}[/]")
        generate_directory_tree(os.path.join(percorso, directory), ramo, limite_file)

    for i, f in enumerate(file):
        if i < limite_file:
            icona = "🖼️" if f.endswith(('.jpg', '.png', '.jpeg')) else "📄"
            tree.add(f"{icona} {f}")
        elif i == limite_file:
            tree.add(f"[italic grey50]... e altri {len(file) - limite_file} file[/]")
            break

    return tree

def plot_animal_samples_in_single_row(base_data_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Cat image
    cat_path = os.path.join(base_data_path, "afhq/train/cat/")
    cat_images = os.listdir(cat_path)
    cat_image = Image.open(os.path.join(cat_path, cat_images[0]))
    axes[0].imshow(cat_image)
    axes[0].set_title("Train Sample [CAT]")
    axes[0].axis('off')

    # Dog image
    dog_path = os.path.join(base_data_path, "afhq/train/dog/")
    dog_images = os.listdir(dog_path)
    dog_image = Image.open(os.path.join(dog_path, dog_images[0]))
    axes[1].imshow(dog_image)
    axes[1].set_title("Train Sample [DOG]")
    axes[1].axis('off')

    # Wild image
    wild_path = os.path.join(base_data_path, "afhq/train/wild/")
    wild_images = os.listdir(wild_path)
    wild_image = Image.open(os.path.join(wild_path, wild_images[0]))
    axes[2].imshow(wild_image)
    axes[2].set_title("Test Sample [WILD]")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def show_sample(img_tensor, label):
    img = img_tensor.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.title(f"Class: {label}, Shape: ({img_tensor.shape[0]},{img_tensor.shape[1]},{img_tensor.shape[2]})")
    plt.axis('off')
    plt.show()

def show_loss(num_epochs, train_metrics, val_metrics):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_metrics['ce_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, val_metrics['ce_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_accuracy(num_epochs, train_metrics, val_metrics):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_metrics['acc'], label='Train Accuracy', marker='o')
    plt.plot(epochs, val_metrics['acc'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_prediction(image, pred, true_label, class2label):
    # convert tensors to numpy
    img = image.cpu().permute(1, 2, 0).numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    fig = plt.figure(figsize=(12, 6))
    plt.imshow(img)

    title = f"Predicted: {class2label[pred.item()]}\nTrue: {class2label[true_label.item()]}"

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()