import os

def organizar_imagens():
    os.system("scripts/organizar_ham10000.py")

def gerar_amostras():
    os.system("scripts/gerar_amostra_classes.py")

def treinar_e_plotar_graficos():
    os.system("scripts/graficos.py")

def treinar_modelo_final():
    os.system("src/training/train.py")

def sair():
    print("Encerrando o programa...")
    exit()

def menu():
    while True:
        print("\n==== MENU ====")
        print("1. Organizar imagens HAM10000")
        print("2. Gerar amostras de cada classe")
        print("3. Treinar modelo (rápido) e gerar gráficos")
        print("4. Treinar modelo final e salvar resultados")
        print("5. Sair")
        opcao = input("Escolha uma opção (1-5): ")

        if opcao == "1":
            organizar_imagens()
        elif opcao == "2":
            gerar_amostras()
        elif opcao == "3":
            treinar_e_plotar_graficos()
        elif opcao == "4":
            treinar_modelo_final()
        elif opcao == "5":
            sair()
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    menu()
