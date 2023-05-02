
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator

filedir = Path(__file__).parent.resolve()

def gen_augmentations(dataset_dir, image_count=16):
    IMAGE_DIMENSIONS = [182,218]
    
    for label in ['AD','CN']:
        # This gets the preprocess_input func i.e apps.resnet.preprocess_input
        train_datagen = ImageDataGenerator(zoom_range=0.05,
                                            width_shift_range=0.05,
                                            height_shift_range=0.05,
                                            vertical_flip=True)
        
        out_path = Path(filedir, '../../out/augmented').resolve()
        out_path.mkdir(parents=True, exist_ok=True)


        train_images = train_datagen.flow_from_directory(
            dataset_dir,
            save_to_dir=Path(filedir, '../../out/augmented').resolve(),
            save_prefix=label+"_aug",
            classes=[label], 
            target_size=IMAGE_DIMENSIONS,
            batch_size=1, 
            class_mode='binary',
        )
        
        # Save the augmented images
        for _ in range(int(image_count/2)):
            next(train_images)


if __name__ == '__main__':
    print("Generating augmentations...")
    print("image_dir: out/preprocessed_datasets/adni_processed/axial_slices")
    gen_augmentations(Path(filedir, '../../out/preprocessed_datasets/adni_processed/axial_slices').resolve(), image_count=16)
    print("Done! Check out/augmented for the results")

