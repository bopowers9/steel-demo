from demo_module.utils import (
    load_images,
    preview_batch,
    configure_data_prefetching,
    plot_performance
)
from demo_module.nn import (
    build_model,
    compile_model,
    fit_model
)

IMG_HEIGHT, IMG_WIDTH = 180, 180
EPOCHS = 10

# Load training and validation sets
train_set, val_set = load_images(
    './data/NEU-DET/IMAGES',
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH
)

# Class labels *should* have been identified from the directory structure
# In production, process should fail gracefully if data are not loaded properly
# For demo purposes, because it's a nice dataset, we'll just assume it is ok
class_names = train_set.class_names
print(f'Identified {len(class_names)} classes: {class_names}')

# Vizualize first batch
preview_batch(train_set.take(1), train_set.class_names)

# Build and compile the model
model = build_model(len(class_names), IMG_HEIGHT, IMG_WIDTH)
model = compile_model(model)

# Fit model
model, history = fit_model(model, train_set, val_set, EPOCHS)

# Plot
plot_performance(history, EPOCHS)
