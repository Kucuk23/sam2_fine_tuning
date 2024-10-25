import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load and resize image
def read_image(image_path):
    img = cv2.imread(image_path)[..., ::-1]  # Convert image to RGB
    r = min(1024 / img.shape[1], 1024 / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    return img

# Load image
image_path = "sema.jpg"
image = read_image(image_path)

# Load the SAM2 model
sam2_checkpoint = "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Set the image for prediction
predictor.set_image(image)

# Randomly sample points within the entire image for prediction
def get_random_points(img_shape, num_points):
    points = np.column_stack((np.random.randint(0, img_shape[1], num_points),
                               np.random.randint(0, img_shape[0], num_points)))
    return points.reshape(-1, 1, 2)

num_samples = 30
input_points = get_random_points(image.shape, num_samples)
point_labels = np.ones((input_points.shape[0], 1))  # Label for each point (1 for foreground)

# Perform prediction
with torch.no_grad():
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=point_labels
    )

# Sort predicted masks by scores
masks = masks[:, 0].astype(bool)
sorted_masks = masks[np.argsort(scores[:, 0])[::-1]]

# Create an overlay for displaying masks
overlay = np.zeros_like(image)

# Generate random colors for each mask and apply them to the overlay
for i, mask in enumerate(sorted_masks):
    if i >= 10:  # Limit to first 10 masks for visibility
        break
    color = np.random.randint(0, 255, size=3).tolist()  # Random color
    overlay[mask] = color  # Apply color to the mask area

# Combine original image and overlay
output_image = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

# Show the output image with predicted masks
cv2.imshow("Predicted Masks", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
