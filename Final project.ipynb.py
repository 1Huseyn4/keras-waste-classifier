

# Task 1
import tensorflow as tf
print(tf.__version__)

# Task 2
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    class_mode='binary',
    seed=42,
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_rows, img_cols)
)

# Task 3
print(len(train_generator))

# Task 4
extract_feat_model.summary()

# Task 5
extract_feat_model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
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
h = fine_tune_history.history
plt.figure()
plt.plot(h['loss'], label='Training loss')
plt.plot(h['val_loss'], label='Validation loss')
plt.title('Fine-Tuned Model: Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.show()

# Task 8
plt.figure()
plt.plot(h['accuracy'], label='Training accuracy')
plt.plot(h['val_accuracy'], label='Validation accuracy')
plt.title('Fine-Tuned Model: Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.show()

# Helper for Tasks 9 and 10
class_names = {v: k for k, v in test_generator.class_indices.items()}
print(class_names)  # yoxlayın: 0 və 1 hansı sinifdir

def plot_test_image(model, generator, index_to_plot, title):
    imgs, labels = generator[0]
    pred = model.predict(imgs)[index_to_plot][0]
    predicted = class_names[int(pred > 0.5)]
    actual = class_names[int(labels[index_to_plot])]
    plt.imshow(imgs[index_to_plot])
    plt.title(f'{title}\nActual: {actual} | Predicted: {predicted}')
    plt.axis('off'); plt.show()

    # Task 9
plot_test_image(extract_feat_model, test_generator, 1, 'Extract Features Model')

# Task 10
plot_test_image(fine_tune_model, test_generator, 1, 'Fine-Tuned Model')