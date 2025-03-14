import os

DATA_FOLDER = "./data"

all_right = True

for idx, filename in enumerate(os.listdir(DATA_FOLDER)):
    if filename.endswith(".txt") and all_right:  # Process only .txt files
        file_path = os.path.join(DATA_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            # Split in paragraphs
            paragraphs = text.split("\n\n")


            minute = 0
            for paragraph in paragraphs:
                minute = (minute + 1) % 60
                paragraph_minute = int(paragraph[3:5])
                all_right = (minute == paragraph_minute)
                if not all_right:
                    print(f'Error in file {filename}')
                    print(f'real minute: {minute}')
                    print(f'txt file minute: {paragraph_minute}')
                    break

    print(f'read file {idx}')

if all_right:
    print("✅ All files clear!")
else:
    print('❌ Not all files clear')
