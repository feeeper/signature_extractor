from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

signatures_folder = 'C:/Users/shirobokov_av/Desktop/Temp/dataset/sign/'
result_signatures_folder = 'C:/Users/shirobokov_av/Desktop/Temp/signatures_augmented/'
for signature_img in os.listdir(signatures_folder):
    img = load_img(signatures_folder + signature_img)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=result_signatures_folder, save_prefix='sign', save_format='jpeg'):
        i += 1
        if i > 20:
            break