import os
import subprocess

# Caminho do script principal
TRAIN_SCRIPT = "src/training/train.py"

def run(command):
    print("\n==============================")
    print(f"🔧 Executando: {command}")
    print("==============================\n")
    subprocess.run(command, shell=True)


def menu():
    while True:
        print("\n==============================")
        print(" 🔥 MENU DE TREINAMENTO IA 🔥")
        print("==============================\n")

        print("1 - Treinar somente MobileNetV2")
        print("2 - Treinar somente VGG16")
        print("3 - Treinar com fine-tuning (10 + 10 épocas)")
        print("4 - Treinar com LR reduzido (20 épocas)")
        print("5 - Rodar pacote completo de testes")
        print("6 - Executar treinamento normal (padrão)")
        print("0 - Sair\n")

        choice = input("Escolha uma opção: ")

        if choice == "1":
            run(f"python {TRAIN_SCRIPT} --model mobilenetv2")

        elif choice == "2":
            run(f"python {TRAIN_SCRIPT} --model vgg16")

        elif choice == "3":
            run(f"python {TRAIN_SCRIPT} --finetune")

        elif choice == "4":
            run(f"python {TRAIN_SCRIPT} --reduced-lr")

        elif choice == "5":
            run(f"python {TRAIN_SCRIPT} --full-test-suite")

        elif choice == "6":
            run(f"python {TRAIN_SCRIPT}")

        elif choice == "0":
            print("\nEncerrando... 👋\n")
            break

        else:
            print("\n⚠️ Opção inválida! Tente novamente.\n")


if __name__ == "__main__":
    menu()
