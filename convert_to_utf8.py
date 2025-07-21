import os
import chardet

folder_path = "data"

if not os.path.exists(folder_path):
    print(f"❌ Mappen '{folder_path}' finnes ikke.")
    exit()

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            raw = f.read()
            result = chardet.detect(raw)
            encoding = result['encoding']
        
        if encoding is None:
            print(f"⚠️ Kunne ikke oppdage tegnsett for {filename}")
            continue
        
        if encoding.lower() != 'utf-8':
            print(f"🔄 Konverterer {filename} fra {encoding} til UTF-8...")
            try:
                content = raw.decode(encoding)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                print(f"❌ Feil ved konvertering av {filename}: {e}")
        else:
            print(f"✅ {filename} er allerede i UTF-8.")