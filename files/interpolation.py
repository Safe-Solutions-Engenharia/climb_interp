from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import warnings
import logging

warnings.filterwarnings("ignore")

#TODO problemas: No caso onde a distância entre os pontos da curva exponencial é muito grande, o fit é esquisito;
class ClimbInterp:
    
    """
    Classe ClimbInterp para interpolação e ajuste de curvas de dados.

    Esta classe é projetada para realizar ajustes de curvas especializados em dados que requerem uma abordagem conservadora,
    priorizando ajustes não negativos para x > 0. Oferece suporte a ajustes exponenciais, lineares e logarítmicos,
    adequando-se automaticamente aos dados fornecidos. 

    Principais Funcionalidades:
    - Realiza ajuste exponencial para a primeira metade dos dados.
    - Escolhe entre ajuste linear e logarítmico para a segunda metade, baseado na distribuição dos dados.
    - Calcula o ponto de intersecção entre os ajustes exponencial e linear/logarítmico.
    - Gera um gráfico visual dos dados e dos ajustes de curva, se necessário.

    A classe é ideal para cenários onde é crucial garantir que a curva ajustada não produza valores negativos
    e onde uma abordagem conservadora é preferível para a interpretação dos dados.
    """
    
    def __init__(self, x_value, y_value, show_graph=False, graph_title=None):

        """
        Construtor da classe ClimbInterp.

        Inicializa uma instância da classe com os conjuntos de dados x e y, além de opções para exibição gráfica.

        :param x_value: Lista ou array de valores x (dados independentes) para a interpolação.
        :param y_value: Lista ou array de valores y (dados dependentes) correspondentes a x_value para a interpolação.
        :param show_graph: Booleano opcional que, se True, habilita a geração de um gráfico dos dados e dos ajustes.
        :param graph_title: String opcional para o título do gráfico, utilizado se show_graph for True.

        A função inicia processando e organizando os dados de entrada, preparando-os para os ajustes de curva subsequentes.
        As propriedades da classe são inicializadas para armazenar informações importantes para os cálculos de ajuste e visualização.
        """

        self.x_value = x_value
        self.y_value = y_value
        self.show_graph = show_graph
        self.graph_title = graph_title

        self.x_exp = None
        self.y_exp = None

        self.popt_exp = None
        self.popt_linear = None
        self.is_linear = False
        self.is_logarithmic = False

        self.intersection = None

        self.curve_fit_exp()
        self.curve_fit_linear()

    def curve_fit_exp(self):

        """
        Realiza o ajuste exponencial dos dados fornecidos.

        Este método aplica um ajuste exponencial na primeira metade dos dados. Utiliza a função auxiliar
        '_get_lower_error' para encontrar o conjunto de pontos que minimiza o erro do ajuste exponencial.
        Os pontos restantes são considerados para um ajuste linear ou logarítmico subsequente.

        O processo envolve:
        - Converter os valores x e y para arrays numpy.
        - Utilizar '_get_lower_error' para otimizar o ajuste exponencial na primeira parte dos dados.
        - Separar os dados que não foram incluídos no ajuste exponencial para uso posterior.
        - Armazenar os parâmetros otimizados do ajuste exponencial e um indicador se o ajuste resultante é linear.

        Após a execução deste método, a instância da classe terá armazenado:
        - 'self.x_exp' e 'self.y_exp': pontos de dados para os quais o ajuste exponencial foi aplicado.
        - 'self.popt_exp': parâmetros otimizados do ajuste exponencial.
        - 'self.is_linear': booleano indicando se o ajuste resultou em uma função linear.
        """

        x = np.array(self.x_value)
        y = np.array(self.y_value)

        x, y, popt, is_linear = self._get_lower_error(x, y)

        new_x = []
        new_y = []
        for x_teste, y_teste in zip(self.x_value, self.y_value):
            if x_teste in x and y_teste in y:
                continue
            else:
                new_x.append(x_teste)
                new_y.append(y_teste)

        self.x_exp = np.concatenate(([x[-1]], new_x))
        self.y_exp = np.concatenate(([y[-1]], new_y))

        self.popt_exp = popt
        self.is_linear = is_linear

    def curve_fit_linear(self):

        """
        Coordena o processo de ajuste linear dos dados, selecionando a abordagem de ajuste apropriada com base no número de pontos.

        Esta função analisa a quantidade de pontos disponíveis após o ajuste exponencial e escolhe o método de ajuste linear mais adequado:
        - Se houver apenas dois pontos (len(self.y_exp) == 2), ela chama '_fit_linear_for_two_points' para um ajuste simples.
        - Se houver mais de dois pontos (len(self.y_exp) > 2), ela usa '_fit_linear_for_multiple_points' para um ajuste mais complexo.
        - Para qualquer outro caso, ela recorre a '_fit_linear_for_single_point', adequado para um único ponto.

        Cada um desses métodos internos trata de casos específicos de ajuste linear, garantindo a melhor aproximação possível para a segunda metade dos dados.
        """

        if len(self.y_exp) == 2:
            self._fit_linear_for_two_points()

        elif len(self.y_exp) > 2:
            self._fit_linear_for_multiple_points()

        else:
            self._fit_linear_for_single_point()

    @staticmethod
    def _linear_fit(x, y):

        """
        Realiza um ajuste linear dos pontos dados (x, y) e calcula o coeficiente de determinação (R²).
        
        A função utiliza programação convexa para maximizar o R², sujeita à restrição de que
        a linha ajustada deve sempre estar acima dos pontos de dados.

        :param x: Lista de valores x (coordenadas dos pontos).
        :param y: Lista de valores y (coordenadas dos pontos).
        :return: Coeficientes da linha de melhor ajuste (inclinação e intercepto) e o valor de R².
                 Retorna (None, None, None) em caso de falha na otimização.
        """

        m = cp.Variable()
        c = cp.Variable()

        constraints = [y[i] <= m * x[i] + c for i in range(len(x))]
        mean_y = sum(y) / len(y)
        tss = cp.sum_squares(y - mean_y)
        rss = cp.sum_squares(y - (m * x + c))
        r_squared = 1 - rss / tss
        objective = cp.Maximize(r_squared)
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve()
        except:
            return None, None, None

        if not r_squared.value:
            r_squared_lin = None
        else:
            r_squared_lin = r_squared.value

        return m.value, c.value, r_squared_lin

    @staticmethod
    def _logarithmic_fit(x, y):

        """
        Realiza um ajuste logarítmico dos pontos dados (x, y) e calcula o coeficiente de determinação (R²).

        O ajuste logarítmico é realizado através da otimização convexa, onde se busca maximizar
        o valor de R². A função assume que os valores de x são positivos, já que o logaritmo
        de valores não positivos não é definido. O problema de otimização é construído para
        maximizar R², sujeito à restrição de que a curva logarítmica ajustada deve sempre
        estar acima dos pontos de dados.

        :param x: Lista de valores x (coordenadas dos pontos).
        :param y: Lista de valores y (coordenadas dos pontos).
        :return: Coeficientes da curva logarítmica de melhor ajuste (inclinação e intercepto) e o valor de R².
                Retorna (None, None, None) em caso de falha na otimização.
        """

        m = cp.Variable()
        c = cp.Variable()

        constraints = [y[i] <= m * cp.log(x[i]) + c for i in range(len(x))]
        mean_y = sum(y) / len(y)
        tss = cp.sum_squares(y - mean_y)
        rss = cp.sum_squares(y - (m * cp.log(np.array(x)) + c))
        r_squared = 1 - rss / tss
        objective = cp.Maximize(r_squared)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
        except:
            return None, None, None

        if not r_squared.value:
            r_squared_log = None
        else:
            r_squared_log = r_squared.value

        return m.value, c.value, r_squared_log

    def _fit_linear_for_single_point(self):

        """
        Realiza o ajuste linear para o caso especial de um único ponto restante após o ajuste exponencial.

        Esta função é utilizada quando há apenas um ponto de dados restante para o ajuste linear. Ela calcula uma linha
        que passa por este ponto e o ponto de origem (0,0), garantindo que a linha ajustada não produza valores negativos.

        O processo envolve:
        - Calcular o coeficiente angular (a) da linha que conecta o ponto de dados ao ponto de origem.
        - Calcular o intercepto (b), que neste caso será sempre 0, pois a linha passa pela origem.
        - Armazenar os parâmetros do ajuste linear (a, b) e a posição x do ponto de dados como o ponto de intersecção.

        Após o cálculo, a função organiza as variáveis para o gráfico e chama '_create_graph' para visualizar o resultado.
        """

        target_y = self.y_exp * 1.1
        a = (abs(target_y - self.y_exp) / abs(1 - self.x_exp)) if abs(1 - self.x_exp) != 0 else 0
        b = self.y_exp - a * self.x_exp

        self.popt_linear = [float(a), float(b)]
        self.intersection = float(self.x_exp)

        x_fit, x_fit_2, y_fit, y_fit_2 = self._arrange_graph_variables()
        self._create_graph(x_fit, y_fit, x_fit_2, y_fit_2)
        
    def _fit_linear_for_two_points(self):
        
        """
        Realiza o ajuste linear ou logarítmico quando existem apenas dois pontos após o ajuste exponencial.

        Esta função é aplicada em um cenário específico onde apenas dois pontos restam para o ajuste linear ou logarítmico. 
        Ela decide o tipo de ajuste baseado na posição relativa desses pontos e no valor do coeficiente R²:

        - Determina os pontos máximos em x e y.
        - Se o ponto máximo em x for menor do que um limiar definido (0.09), realiza um ajuste linear simplificado.
        - Caso contrário, compara os ajustes linear e logarítmico, escolhendo aquele com o maior R².
        - Armazena os parâmetros do ajuste escolhido e atualiza 'self.is_logarithmic' conforme necessário.

        Finalmente, organiza as variáveis para o gráfico e chama '_create_graph' para exibir os resultados.
        """

        y_max_index = np.argmax(self.y_exp)
        x_max_index = np.argmax(self.x_exp)
        x = [self.x_exp[0], self.x_exp[y_max_index]]
        y = [self.y_exp[0], self.y_exp[y_max_index]]

        if self.x_exp[x_max_index] < 0.09:

            target_y = self.y_exp[x_max_index] * 1.1
            m_fit = (abs(target_y - self.y_exp[x_max_index]) / abs(1 - self.x_exp[x_max_index]))
            c_fit = self.y_exp[x_max_index] - m_fit * self.x_exp[x_max_index]

            popt = [float(m_fit), float(c_fit)]
        else:

            m_linear, c_linear, r_squared_linear = self._linear_fit(x, y)
            m_logarithmic, c_logarithmic, r_squared_logarithmic = self._logarithmic_fit(x, y)

            if r_squared_linear and r_squared_logarithmic:
                if r_squared_linear >= r_squared_logarithmic:
                    popt = [m_linear, c_linear]
                else:
                    popt = [m_logarithmic, c_logarithmic]
                    self.is_logarithmic = True
            elif r_squared_linear:
                popt = [m_linear, c_linear]
            elif r_squared_logarithmic:
                popt = [m_logarithmic, c_logarithmic]
                self.is_logarithmic = True
            else:
                popt = [None, None]

            if not popt[0] and not popt[1]:
                self.is_logarithmic = False

                popt, pcov = optimize.curve_fit(self._linear, x, y)

        self.popt_linear = [float(popt[0]), float(popt[1])]
        self.intersection = float(self.x_exp[0])

        x_fit, x_fit_2, y_fit, y_fit_2 = self._arrange_graph_variables()
        self._create_graph(x_fit, y_fit, x_fit_2, y_fit_2)
    
    def _fit_linear_for_multiple_points(self):

        """
        Realiza o ajuste linear ou logarítmico para um conjunto de múltiplos pontos.

        Esta função é usada quando mais de dois pontos estão disponíveis para ajuste após o processo de ajuste exponencial.
        Ela decide qual ajuste aplicar (linear ou logarítmico) com base na distribuição dos pontos e no valor de R²:

        - Primeiro, verifica se o ponto máximo em x é menor do que um limiar definido (0.09).
          Se for, realiza um ajuste linear simplificado.
        - Caso contrário, compara os ajustes linear e logarítmico, calculando o R² para cada um e escolhendo o que tiver maior R².
        - Se nenhum ajuste for adequado, tenta encontrar o melhor ajuste linear possível percorrendo combinações de pontos.
        - Atualiza 'self.is_logarithmic' conforme necessário e armazena os parâmetros do ajuste escolhido.

        Após determinar o melhor ajuste, a função organiza as variáveis para o gráfico e chama '_create_graph' para visualizar os resultados.
        """

        max_index = np.argmax(self.x_exp)
        if self.x_exp[max_index] < 0.09:

            target_y = self.y_exp[max_index] * 1.1
            m_fit = (abs(target_y - self.y_exp[max_index]) / abs(1 - self.x_exp[max_index]))
            c_fit = self.y_exp[max_index] - m_fit * self.x_exp[max_index]

        else:
            m_linear, c_linear, r_squared_linear = self._linear_fit(self.x_exp, self.y_exp)
            m_logarithmic, c_logarithmic, r_squared_logarithmic = self._logarithmic_fit(self.x_exp, self.y_exp)

            if r_squared_linear and r_squared_logarithmic:
                if r_squared_linear >= r_squared_logarithmic:
                    m_fit = m_linear
                    c_fit = c_linear
                else:
                    m_fit = m_logarithmic
                    c_fit = c_logarithmic
                    self.is_logarithmic = True
            elif r_squared_linear:
                m_fit = m_linear
                c_fit = c_linear
            elif r_squared_logarithmic:
                m_fit = m_logarithmic
                c_fit = c_logarithmic
                self.is_logarithmic = True
            else:
                m_fit = None
                c_fit = None

            if not m_fit and not c_fit and m_fit != 0 and c_fit != 0:

                self.is_logarithmic = False

                ar_x = np.array(self.x_exp)
                ar_y = np.array(self.y_exp)
                # Sort the data points by ar_x-values
                sorted_indices = np.argsort(ar_x)
                ar_x_sorted = ar_x[sorted_indices]
                ar_y_sorted = ar_y[sorted_indices]

                best_fit_r_squared = -999
                m_fit = None
                c_fit = None

                for i in range(len(ar_x_sorted) - 1):
                    for j in range(i + 1, len(ar_x_sorted)):
                        ar_x1, ar_x2 = ar_x_sorted[i], ar_x_sorted[j]
                        ar_y1, ar_y2 = ar_y_sorted[i], ar_y_sorted[j]

                        m = (ar_y2 - ar_y1) / (ar_x2 - ar_x1)
                        b = ar_y1 - m * ar_x1

                        line = m * ar_x + b

                        if np.all(np.round(ar_y, 0) <= np.round(line, 0)):
                            fitted_y_ar = self._linear(ar_x_sorted, m, b)
                            r_squared = _r_squared(ar_y_sorted, fitted_y_ar)

                            if r_squared > best_fit_r_squared:
                                best_fit_r_squared = r_squared
                                m_fit = m
                                c_fit = b

        self.popt_linear = [float(m_fit), float(c_fit)]
        self.intersection = float(self.x_exp[0])

        x_fit, x_fit_2, y_fit, y_fit_2 = self._arrange_graph_variables()
        self._create_graph(x_fit, y_fit, x_fit_2, y_fit_2)

    def _get_lower_error(self, x, y):

        """
        Determina o ajuste de curva com menor erro para um conjunto de pontos.

        :param x: Lista de valores x.
        :param y: Lista de valores y.
        :return: Conjunto de pontos otimizado, parâmetros da curva e se a curva é uma linha.
        """

        if len(x) == 1:
            return self._fit_curve_for_single_point(x, y)

        for index in range(len(x) - 1):
            x_fake, y_fake = x[:index + 2], y[:index + 2]
            x_fake, y_fake, popt, is_line = self._fit_curve_and_evaluate(x_fake, y_fake)

            if len(x) == 2:
                return x_fake, y_fake, popt, is_line

            # Tenta incluir o próximo ponto e avaliar novamente
            if index + 3 <= len(x):
                x_new, y_new = x[:index + 3], y[:index + 3]
                x_new, y_new, popt_new, is_line_new = self._fit_curve_and_evaluate(x_new, y_new)

                if not is_line_new and not is_line:
                    return x_fake, y_fake, popt, False

        return x, y, popt, False

    def _fit_curve_for_single_point(self, x, y):

        """
        Realiza um ajuste de curva para um único ponto.

        :param x: Lista com um valor x.
        :param y: Lista com um valor y.
        :return: x, y, parâmetros otimizados e um indicador se é uma linha.
        """

        popt, _ = optimize.curve_fit(Functions.exception_linear, [0, x[0]], [0, y[0]])
        return x, y, popt, True

    def _fit_curve_and_evaluate(self, x, y):

        """
        Realiza o ajuste da curva e avalia a qualidade do ajuste usando R².

        :param x: Lista de valores x.
        :param y: Lista de valores y.
        :return: x, y, parâmetros otimizados e um indicador se é uma linha, baseado no R².
        """

        try:
            popt, _ = optimize.curve_fit(Functions.exp, x, y, maxfev=10000)
            fitted_y = Functions.exp(x, *popt)
            r_squared_value = Functions.r_squared(y, fitted_y)
            return x, y, popt, r_squared_value < 0
        except Exception as e:
            logging.warning(e)
            return x, y, None, False

    def _arrange_graph_variables(self):

        """
        Prepara as variáveis para plotagem dos gráficos, incluindo os pontos para os ajustes exponencial e linear/logarítmico.
        
        - Calcula pontos de ajuste exponencial (x_fit_2, y_fit_2) para o intervalo inicial dos dados.
        - Calcula pontos de ajuste linear ou logarítmico (x_fit, y_fit) para o intervalo final dos dados.
        - A escolha entre ajuste linear e logarítmico depende do valor de 'is_logarithmic'.
        - O ajuste exponencial é modificado se 'is_linear' for True, usando um modelo linear especial.
        
        :return: Quatro listas contendo pontos x e y para os ajustes exponencial e linear/logarítmico.
        """

        # Fit exponencial
        x_fit_2 = np.linspace(0, self.x_exp[0], 100)

        if not self.is_linear:
            y_fit_2 = Functions.exp(x_fit_2, *self.popt_exp)
        else:
            y_fit_2 = Functions.exception_linear(x_fit_2, *self.popt_exp)

        # Fit linear/logarítmico
        x_fit = np.linspace(self.x_exp[0], self.x_exp[-1], 100)

        if self.is_logarithmic:
            y_fit = Functions.logarithmic(x_fit, *self.popt_linear)
        else:
            y_fit = Functions.linear(x_fit, *self.popt_linear)

        return x_fit, x_fit_2, y_fit, y_fit_2

    def _create_graph(self, x_fit, y_fit, x_fit_2, y_fit_2):

        """
        Cria um gráfico para visualizar os dados e seus ajustes.

        Esta função plota os dados originais e os ajustes de curva exponencial e linear/logarítmico.
        - Os dados originais são plotados como pontos dispersos.
        - O ajuste exponencial é plotado como uma linha tracejada azul.
        - O ajuste linear ou logarítmico é plotado como uma linha contínua, verde para logarítmico e vermelha para linear.
        - As configurações do gráfico incluem limites de eixos, legendas e título.
        - O gráfico é exibido apenas se 'show_graph' for True.

        :param x_fit: Pontos x para o ajuste linear/logarítmico.
        :param y_fit: Pontos y para o ajuste linear/logarítmico.
        :param x_fit_2: Pontos x para o ajuste exponencial.
        :param y_fit_2: Pontos y para o ajuste exponencial.
        """
        if self.show_graph:
            plt.figure()
            plt.scatter(self.x_value, self.y_value, label='Dados')
            if self.is_logarithmic:
                plt.plot(x_fit, y_fit, 'g', label='Log')
            else:
                plt.plot(x_fit, y_fit, 'r', label='Reta')

            plt.plot(x_fit_2, y_fit_2, '--b', label='Exponencial')
            plt.ylim(bottom=0)
            plt.xlim(left=0)
            plt.xlim(right=self.x_exp[-1])
            plt.legend()
            plt.grid(True)
            plt.title(self.graph_title)
            plt.show()

    def get_important_variables(self):

        """
        Retorna as variáveis importantes obtidas após o processo de interpolação.

        Esta função facilita o acesso aos parâmetros chave e informações de estado resultantes dos ajustes de curva.

        :return: Uma tupla contendo:
                - self.popt_exp: Parâmetros otimizados do ajuste exponencial.
                - self.intersection: Ponto de intersecção entre os ajustes exponencial e linear/logarítmico.
                - self.popt_linear: Parâmetros otimizados do ajuste linear ou logarítmico.
                - self.is_linear: Indicador booleano de se o ajuste foi linear (True) ou exponencial (False).
                - self.is_logarithmic: Indicador booleano de se o ajuste foi logarítmico (True) ou linear (False).
        """

        return self.popt_exp, self.intersection, self.popt_linear, self.is_linear, self.is_logarithmic

class Functions:

    """
    Classe Functions para agrupar funções de interpolação e ajuste de curvas.

    Esta classe contém funções utilizadas para realizar diferentes tipos de ajustes em conjuntos de dados, como ajustes lineares, 
    logarítmicos e exponenciais. As funções são projetadas para serem genéricas e reutilizáveis, facilitando sua aplicação em 
    diferentes contextos onde ajustes de curva são necessários.

    Principais características:
    - Funções para realizar ajustes de curva específicos (linear, logarítmico, exponencial).
    - Funções auxiliares para calcular métricas como o coeficiente de determinação (R²).
    - Métodos para preparar e organizar dados para ajustes de curva.
    """

    @staticmethod
    def exp(x, a, b):
        return a * np.exp(b*x)

    @staticmethod
    def logarithmic(x, a, b):
        return a * np.log(x) + b

    @staticmethod
    def linear(x, a, b):
        return a * x + b

    @staticmethod
    def exception_linear(x, a):
        return x * a

    @staticmethod
    def r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)