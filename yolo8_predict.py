from ultralytics import YOLO

# Load a model
model = YOLO("jetracer\\VERY_BEST.pt")  # pretrained YOLO11n model
images_path = 'jetracer\\test_images'

# Run batched inference on a list of images
results = model(images_path)  # return a list of Results objects

# Process results list
for result in results:
    result.show()  # display to screen