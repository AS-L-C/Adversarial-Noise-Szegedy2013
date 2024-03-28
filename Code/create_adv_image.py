import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.functional import to_pil_image


def get_model():
    """Retrieves ResNet18, its image preprocessing function, and the ImageNet labels"""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    prep_fcn = weights.transforms()
    categories = weights.meta["categories"]
    return model, prep_fcn, categories


def to_pil(img_tensor):
    """Converts tensors to PIL images"""
    _, prep_fcn, _ = get_model()
    stds = torch.tensor(prep_fcn.std)
    means = torch.tensor(prep_fcn.mean)
    img = img_tensor * stds.view(3, 1, 1) + means.view(3, 1, 1)
    return to_pil_image(img)


def loss_fcn(logits, target_label, r, c=0.4):
    """Computes the loss to minimize to build adversarial noise according to Szegedy et al. 2013"""
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(logits, target_label) + c * torch.norm(r)
    return loss


class ResNetPert(nn.Module):
    def __init__(self, inp_shape, device):
        super(ResNetPert, self).__init__()
        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        self.model, self.prep_fcn, self.categories = get_model()
        self.model.to(self.device)
        self.freeze_pars()
        self.r = torch.nn.Parameter(
            torch.randn((inp_shape), requires_grad=True, device=self.device)
        )

    def freeze_pars(self):
        for param in self.model.parameters():
            param.requires_grad = False
        return

    def forward(self, x):
        logits = self.model(x + self.r)
        return logits


def get_label(logits):
    """Computes the label and the confidence from a model logits"""
    confidence = logits.squeeze(0).softmax(0)
    label = confidence.argmax()
    return label, confidence[label].item()


def train(model, img, target_label, n_epochs):
    """Minimizes the loss to compute the noise"""
    target_label = torch.tensor(target_label, device=model.device).unsqueeze(0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.005,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    )
    print("Training starting...")
    for epoch in range(n_epochs):
        logits = model(img)
        loss = loss_fcn(logits, target_label, model.r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print summary
        print(
            "Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, n_epochs, loss.item()),
            end="",
        )

        # Compute label and confidence
        if ((epoch + 1) % 10 == 0) or (epoch == n_epochs - 1):
            lab, conf = get_label(model(img))
            print(f", Label (confidence) : {lab} ({conf:.4f})")
        else:
            print()
    print("Training ended.")
    return lab, conf


def prep_img(img):
    """Preprocesses an image to be compatible with ResNet18"""
    _, prep_fcn, _ = get_model()
    return prep_fcn(img)


def print_init_info(true_label, init_label, init_confid):
    """Prints useful information before training"""
    conf_str = "(confidence)"
    print(f"True Label {' '*len(conf_str)} : {true_label}")
    print(f"Init Label {conf_str} : {init_label} ({init_confid:4f})")
    return


def compute_perturbation(img, true_label, target_label, n_epochs, device):
    """Computes the additive noise that tricks the model to predict the target_label"""
    img = img.unsqueeze(0).to(device)
    model = ResNetPert(inp_shape=img.shape, device=device)
    init_label, init_confid = get_label(model.model(img))
    print_init_info(true_label, init_label, init_confid)
    final_label, final_confid = train(model, img, target_label, n_epochs)
    pert_img = img + model.r
    info = {
        "init_label": init_label.item(),
        "init_confid": init_confid,
        "final_label": final_label.item(),
        "final_confid": final_confid,
    }
    return pert_img.squeeze(), model.r.squeeze(), info


def create_summary_plot(
    orig,
    pert,
    noise,
    info,
    summary_img_fld,
    show=False,
    save=True,
):
    """Creates a summary plot"""
    classes = info["classes"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Original image
    axes[0].imshow(orig)
    axes[0].set_title(
        f"Original (prediction: {classes[info['init_label']]}, confidence: {info['init_confid']:.3f})"
    )
    axes[0].axis("off")

    # Perturbed image
    axes[1].imshow(pert)
    axes[1].set_title(
        f"Perturbed (prediction: {classes[info['final_label']]}, confidence: {info['final_confid']:.3f})"
    )
    axes[1].axis("off")

    # Noise
    axes[2].imshow(noise)
    axes[2].set_title("Added Noise")
    axes[2].axis("off")

    # Suptitle
    true_label = classes[info["true_label"]] if info["true_label"] != -1 else "NA"
    tar_label = classes[info["target_label"]]
    fig.suptitle(
        f"File: {info['file_name']}, True Label: {true_label}, Target Label: {tar_label}",
        fontsize=14,
        y=1,
    )

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    if save:
        plt.savefig(summary_img_fld + "/" + info["file_name"])

    # Show the plot
    if show:
        plt.show()
    return


def perturb_image(
    target_label,
    img_fld,
    img_name,
    pert_img_fld,
    summary_img_fld,
    n_epochs,
    device,
):
    """Loads the selected image, creates adversarial noise, generates summary plots"""
    directory_path = Path(img_fld)

    if img_name == "all":
        files = directory_path.glob("*")
    else:
        files = [directory_path / img_name]
    _, _, categories = get_model()
    for file in files:
        # Load and preprocess image
        img = prep_img(Image.open(file))
        match = re.search(r"_([0-9]+)\.jpg$", file.name)
        true_label = int(match.group(1)) if match else -1

        # Create perturbed image
        img_pert, noise, info = compute_perturbation(
            img,
            true_label,
            target_label,
            n_epochs,
            device,
        )

        # Store perturbed image
        img_pert = to_pil(img_pert.to(torch.device("cpu")))
        img = to_pil(img.to(torch.device("cpu")))
        noise = to_pil(noise.to(torch.device("cpu")))

        img_pert.save(pert_img_fld + "/" + file.name)

        # Create summary plots
        img_info = {
            "file_name": file.name,
            "true_label": true_label,
            "target_label": target_label,
            "classes": categories,
            **info,
        }
        create_summary_plot(
            orig=img,
            pert=img_pert,
            noise=noise,
            info=img_info,
            summary_img_fld=summary_img_fld,
            show=False if img_name == "all" else True,
        )

    return


def parse_arguments(parser):
    """Parses input arguments"""
    parser.add_argument(
        "--target_label",
        default=0,
        help="Index of the target label",
    )

    parser.add_argument(
        "--img_fld",
        default="../Original_Images",
        help="Path to folder containing image to perturb",
    )

    parser.add_argument(
        "--img_name",
        default="validation_0_label_91.jpg",  # "all",  # "validation_0_label_91.jpg",
        help="Name of image to perturb (set to 'all' to create perturbed versions of all images in img_fld)",
    )

    parser.add_argument(
        "--pert_img_fld",
        default="../Perturbed_Images",
        help="Path to folder that will contain the perturbed image",
    )

    parser.add_argument(
        "--summary_img_fld",
        default="../Summary_Plots",
        help="Path to folder that will store the summary plots",
    )

    parser.add_argument(
        "--n_epochs",
        default=2000,
        help="The number of epochs to optimize the cost function for",
        type=int,
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="The device to perform the optimization on",
    )

    args = parser.parse_args()
    return vars(args)


def main():
    parser = argparse.ArgumentParser(
        description="Perturbs an input image according to Szegedy et al. 2013."
    )
    args = parse_arguments(parser)
    perturb_image(**args)


if __name__ == "__main__":
    main()
