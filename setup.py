import os
import sys
import shutil
import glob


if __name__ == '__main__':
    # Get the path to the database directory from args
    if len(sys.argv) == 1:
        print('You must provide an absolute path to the data directory as an argument!')
        sys.exit(0)
    data_path = sys.argv[1]

    # Change working directory to the provided path
    os.chdir(data_path)

    # Class names in the NEU surface defect database
    class_names = [
        'crazing',
        'inclusion',
        'patches',
        'pitted_surface',
        'rolled-in_scale',
        'scratches'
    ]

    # Setup each class individually
    for class_name in class_names:
        # Folder name should use CAPS and underscores, for consistency
        folder_name = class_name.upper().replace('-', '_')
        # Only proceed if folder does not already exist
        if not os.path.isdir(folder_name):
            print(f'Setting up folder: {folder_name}')
            os.mkdir(folder_name)
            # Move all files for this class into the new folder
            for file_path in glob.glob(f'{class_name}_*'):
                shutil.move(file_path, folder_name)

    print('Setup is complete')
