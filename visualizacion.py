
import os 
import random
import matplotlib.pyplot as plt
import torch

from model import Net, get_dataloaders, load_model


# Dataset 

def explore_dataset(trainset) -> None:

    # Muestra de imagen
    for data in trainset:
        X, y = data
        break

    plt.figure(figsize=(3, 3))
    plt.imshow(X[0].view(28, 28), cmap="gray")
    plt.title(f"Etiqueta: {y[0].item()}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Balance de clases
    total = 0
    counter = {i: 0 for i in range(10)}

    for X_batch, y_batch in trainset:
        for label in y_batch:
            counter[int(label)] += 1
            total += 1

    print("\nBalance de clases en entrenamiento:")
    for digit, count in counter.items():
        print(f"  {digit}: {count:>5}  ({count / total * 100:.2f} %)")


# Predicciones aleatorias 

def show_random_predictions(model: Net, test_loader, num_images: int = 5) -> None:
    
    model.eval()

    all_images, all_labels = [], []
    for X, y in test_loader:
        all_images.append(X)
        all_labels.append(y)

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    indices = random.sample(range(len(all_images)), min(num_images, len(all_images)))

    cols = min(num_images, 5)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = [axes] if num_images == 1 else axes.flatten()

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            image = all_images[idx]
            true_label = all_labels[idx].item()
            output = model(image.view(-1, 784))
            pred_label = torch.argmax(output[0]).item()

            color = "green" if pred_label == true_label else "red"
            ax.imshow(image.view(28, 28), cmap="gray")
            ax.set_title(f"Pred: {pred_label}  Real: {true_label}", color=color, fontsize=9)
            ax.axis("off")

    # Ocultar ejes sobrantes
    for ax in axes[len(indices):]:
        ax.set_visible(False)

    plt.suptitle("Predicciones aleatorias  (verde = correcto, rojo = incorrecto)", fontsize=10)
    plt.tight_layout()
    plt.show()
    print(f"Se mostraron {len(indices)} imágenes.")


#  Errores de clasificación 

def show_misclassified(
    model: Net,
    test_loader,
    num_errors: int = 5,
    save_dir: str = "misclassified_images",
) -> None:
    """
    Muestra y guarda imágenes donde el modelo se equivocó.

    Args:
        model:      Red neuronal entrenada.
        test_loader: DataLoader del conjunto de prueba.
        num_errors: Cuántos errores mostrar/guardar.
        save_dir:   Carpeta donde guardar las imágenes fallidas.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    errors = []
    with torch.no_grad():
        for X, y in test_loader:
            for i in range(X.shape[0]):
                if len(errors) >= num_errors:
                    break
                image = X[i]
                true_label = y[i].item()
                pred_label = torch.argmax(model(image.view(-1, 784))[0]).item()
                if pred_label != true_label:
                    errors.append((image, true_label, pred_label))
            if len(errors) >= num_errors:
                break

    if not errors:
        print("¡Sin errores en las muestras revisadas!")
        return

    cols = min(len(errors), 5)
    rows = (len(errors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = [axes] if len(errors) == 1 else axes.flatten()

    for ax, (image, true_label, pred_label) in zip(axes, errors):
        ax.imshow(image.view(28, 28), cmap="gray")
        ax.set_title(f"Pred: {pred_label}  Real: {true_label}", color="red", fontsize=9)
        ax.axis("off")

        fname = os.path.join(save_dir, f"pred{pred_label}_real{true_label}.png")
        plt.imsave(fname, image.view(28, 28).cpu().numpy(), cmap="gray")

    for ax in axes[len(errors):]:
        ax.set_visible(False)

    plt.suptitle("Predicciones incorrectas", fontsize=10)
    plt.tight_layout()
    plt.show()
    print(f"Se mostraron y guardaron {len(errors)} errores en '{save_dir}/'.")


# Script principal 

if __name__ == "__main__":
    trainset, testset = get_dataloaders()

    print("=== Exploración del dataset ===")
    explore_dataset(trainset)

    print("\n=== Cargando modelo ===")
    net = load_model("mnist_nn_model.pth")

    print("\n=== Predicciones aleatorias ===")
    show_random_predictions(net, testset, num_images=5)

    print("\n=== Errores de clasificación ===")
    show_misclassified(net, testset, num_errors=5)