import os

DATA_FOLDER = "./data"

all_right = True
false_file = ""

for idx, filename in enumerate(os.listdir(DATA_FOLDER)):
    if filename.endswith(".txt") and all_right:  # Process only .txt files
        file_path = os.path.join(DATA_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            text = text.strip('\n')
            text = text.strip()
            # Split in paragraphs
            paragraphs = text.split("\n\n")

            start_minute = int(paragraphs[0][3:5])
            minute = 0 if start_minute==1 else -1
            for paragraph in paragraphs:
                minute = (minute + 1) % 60
                paragraph_minute = paragraph[3:5]
                if not paragraph_minute.isdigit():
                    print(filename)
                    print(minute)
                    print(paragraph)

                paragraph_minute = int(paragraph[3:5])
                all_right = (minute == paragraph_minute)
                if not all_right:
                    print(f'Error in file {filename}')
                    print(f'real minute: {minute}')
                    print(f'txt file minute: {paragraph_minute}')
                    false_file = filename
                    break

        print(f'read file {idx}')

    if not all_right:
        break

if all_right:
    print("✅ All files clear!")
else:
    print('❌ Not all files clear')
    print(f'Error in file {false_file}')
