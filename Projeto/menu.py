import os
import subprocess

TRAIN_SCRIPT   = "src/training/train.py"
PREDICT_SCRIPT = "predict.py"


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

        print("--- Treinamento ---")
        print("1 - Treinar somente MobileNetV2")
        print("2 - Treinar somente VGG16")
        print("3 - Treinar somente EfficientNetB0")
        print("4 - Treinar com fine-tuning (10 + 10 épocas)")
        print("5 - Treinar com LR reduzido (20 épocas)")
        print("6 - Rodar pacote completo (VGG16 + MobileNetV2 + EfficientNetB0)")
        print("7 - Executar treinamento padrão (VGG16 + MobileNetV2)")
        print()
        print("--- Inferência ---")
        print("8 - Classificar nova imagem (+ Grad-CAM)")
        print()
        print("0 - Sair\n")

        choice = input("Escolha uma opção: ").strip()

        if choice == "1":
            run(f"python {TRAIN_SCRIPT} --model mobilenetv2")

        elif choice == "2":
            run(f"python {TRAIN_SCRIPT} --model vgg16")

        elif choice == "3":
            run(f"python {TRAIN_SCRIPT} --model efficientnetb0")

        elif choice == "4":
            run(f"python {TRAIN_SCRIPT} --finetune")

        elif choice == "5":
            run(f"python {TRAIN_SCRIPT} --reduced-lr")

        elif choice == "6":
            run(f"python {TRAIN_SCRIPT} --full-test-suite")

        elif choice == "7":
            run(f"python {TRAIN_SCRIPT}")

        elif choice == "8":
            img  = input("Caminho da imagem (.jpg ou .png): ").strip()
            mdl  = input("Caminho do modelo (.h5): ").strip()
            arch = input("Arquitetura (vgg16 / mobilenetv2 / efficientnetb0): ").strip()
            run(f'python {PREDICT_SCRIPT} --image "{img}" --model "{mdl}" --arch {arch}')

        elif choice == "0":
            print("\nEncerrando... 👋\n")
            break

        else:
            print("\n⚠️ Opção inválida! Tente novamente.\n")


if __name__ == "__main__":
    menu()
