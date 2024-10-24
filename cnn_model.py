
import matplotlib.pyplot as plt
obj = pp.preprocess_data()
dir_path = "train"
retina_df, train, labels = obj.preprocess(dir_path)
# data augmentation
tr_gen, tt_gen, va_gen = obj.generate_train_test_images(retina_df, train,
labels)
print("Batch Normalization")
cnn_b=cnn_batch.DeepANN().cnn_add_batch_Norm()
cnn_history = cnn_b.fit(tr_gen,validation_data=va_gen,epochs=5)
cnn_loss,cnn_acc=cnn_b.evaluate(tt_gen)
print(f"test accuracy: {cnn_acc}")
cnn_b.save("cnn_batch.keras")
print("the ann Architecture")
print(cnn_b.summary())
# visualizing results
plt.figure(figsize=(10, 6)y)
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['loss'], label='Train loss')
plt.plot(cnn_history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['accuracy'], label='Train accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
