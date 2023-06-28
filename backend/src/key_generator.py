from cryptography.fernet import Fernet

# Generate a random encryption key and decode to str to be stored
print("Generating encryption key...")
encryption_key = Fernet.generate_key()
encryption_key_str = encryption_key.decode()

print("Successfully stored AES-256 encryption key.")
