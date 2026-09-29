# Task 1
import tensorflow as tf
print(tf.__version__)

# Task 2
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    class_mode='binary',
    seed=seed_num,          # notebook-dakı dəyişən (yoxlayın)
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_rows, img_cols)
)

# Task 3
print(len(train_generator))

# Task 4
extract_feat_model.summary()
fine_tune_model.summary()

# Task 5
from tensorflow.keras import optimizers
extract_feat_model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)
fine_tune_model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-5),
    metrics=['accuracy']
)

# Task 6
import matplotlib.pyplot as plt
h = extract_feat_history.history
plt.figure()
plt.plot(h['accuracy'], label='Training accuracy')
plt.plot(h['val_accuracy'], label='Validation accuracy')
plt.title('Extract Features Model: Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.show()

# Task 7
ft = fine_tune_history.history
plt.figure()
plt.plot(ft['loss'], label='Training loss')
plt.plot(ft['val_loss'], label='Validation loss')
plt.title('Fine-Tuned Model: Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.show()

# Task 8
ft = fine_tune_history.history
plt.figure()
plt.plot(ft['accuracy'], label='Training accuracy')
plt.plot(ft['val_accuracy'], label='Validation accuracy')
plt.title('Fine-Tuned Model: Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.show()

# Task 9 & 10 üçün köməkçi funksiya
import numpy as np

test_generator.reset()
imgs, labels = next(test_generator)          # ilk batch
class_names = {v: k for k, v in test_generator.class_indices.items()}
print(class_names)

def plot_test_image(model, index_to_plot, title):
    preds = model.predict(imgs, verbose=0)
    if preds.shape[-1] == 1:                  # sigmoid
        predicted_idx = int(preds[index_to_plot][0] > 0.5)
    else:                                     # softmax
        predicted_idx = int(np.argmax(preds[index_to_plot]))
    actual_idx = int(labels[index_to_plot])
    plt.imshow(imgs[index_to_plot])
    plt.title(f'{title}\nActual: {class_names[actual_idx]} | Predicted: {class_names[predicted_idx]}')
    plt.axis('off')
    plt.show()

# Task 9
plot_test_image(extract_feat_model, 1, 'Extract Features Model')

# Task 10
plot_test_image(fine_tune_model, 1, 'Fine-Tuned Model')