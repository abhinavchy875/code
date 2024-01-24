class Contact:
    def _init_(self, name, phone):
        self.name = name
        self.phone = phone

class Phonebook:
    def _init_(self):
        self.contacts = {}

    def add_contact(self, contact):
        if contact.name in self.contacts:
            print(f"Contact {contact.name} already exists in the phonebook.")
        else:
            self.contacts[contact.name] = contact

    def remove_contact(self, name):
        if name in self.contacts:
            del self.contacts[name]
        else:
            print(f"Contact {name} does not exist in the phonebook.")

    def search_contact(self, name):
        if name in self.contacts:
            contact = self.contacts[name]
            print(f"Name: {contact.name}, Phone: {contact.phone}")
        else:
            print(f"Contact {name} does not exist in the phonebook.")

def main():
    phonebook = Phonebook()

    while True:
        print("\nPhonebook Options:")
        print("1. Add Contact")
        print("2. Remove Contact")
        print("3. Search Contact")
        print("4. Exit")

        option = int(input("Enter the option number: "))

        if option == 1:
            name = input("Enter the contact name: ")
            phone = input("Enter the contact phone number: ")
            contact = Contact(name, phone)
            phonebook.add_contact(contact)
        elif option == 2:
            name = input("Enter the contact name: ")
            phonebook.remove_contact(name)
        elif option == 3:
            name = input("Enter the contact name: ")
            phonebook.search_contact(name)
        elif option == 4:
            break
        else:
            print("Invalid option. Please try again.")
