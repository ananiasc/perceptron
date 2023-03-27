import random

class Perceptron(object):
    # Inicializando as variaveis permitindo o usuario poder alterar os valores;
    def __init__(self, eta=0.01, vies=0.5, peso_do_vies = 0, n_iter=1000):
        # Taxa de Aprendizagem
        self.eta = eta
        # Vies
        self.vies = vies
        # Peso do vies
        self.peso_do_vies = peso_do_vies
        # Numero de iteracoes EPOCAS
        self.n_iter = n_iter

    def fit(self, X_treino, y_treino):
        n_epocas = 0
        self.w_ =  []
        for i in range(X_treino.shape[1]):
            self.w_.append(random.uniform(0, 1))
        print(self.w_)
        
        while True:
            erro = False
            for xi, alvo in zip(X_treino, y_treino):
                u = self.peso_do_vies * self.vies # w0 * limiar
                for j in range(len(self.w_)):
                    u += self.w_[j] * xi[j]
                # atribuindo valor de saída y baseado em step function
                # valores de u >=0 recebe 1, caso contrario recebe 0
                y = 1 if u >=0 else -1
                e = alvo - y;
                
                if e != 0:
                    for j in range(len(self.w_)):
                        self.w_[j] += self.eta * e * xi[j] # ajuste dos pesos
                    erro = True
                
            n_epocas += 1 # acrescenta a epoca
            if erro == False or n_epocas >= self.n_iter:
                    break
    
    def predict(self, X_teste):
        Y_previsto = []
        
        for xi in X_teste:
            u = self.peso_do_vies * self.vies # w0 * limiar
            for j in range(len(self.w_)):
                u += self.w_[j] * xi[j]
                
            y = 1 if u >=0 else -1
            Y_previsto.append(y)
        
        return Y_previsto

    def getW(self):
        return self.w_