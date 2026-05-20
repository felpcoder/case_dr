
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
 
 

class Plots:
    
    """
    Responsável por renderizar gráficos de séries temporais agregadas.
 
    Desacoplada da lógica de análise para permitir reuso em qualquer
    contexto onde se tenha uma pd.Series indexada por período — seja
    contagem, média, mediana, soma ou qualquer outra métrica.
 
    Exemplo de reuso futuro
    -----------------------
    >>> media_mensal = (
    ...     df.assign(periodo=pd.to_datetime(df["data_contrato"])
    ...                         .dt.to_period("M").astype(str))
    ...       .groupby("periodo")["valor_contrato"]
    ...       .mean()
    ...       .sort_index()
    ... )
    >>> plotter = DatePlotter()
    >>> plotter.bar(
    ...     series=media_mensal,
    ...     title="Valor médio de contrato por mês",
    ...     xlabel="Ano-Mês",
    ...     ylabel="Valor médio (R$)",
    ...     color="#DD8452",
    ... )
    """
 
    DEFAULT_COLOR = "#4C72B0"
    DEFAULT_FIGSIZE = (14, 4)
    DEFAULT_ROTATE_X = 45
 
    def bar(
        self,
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        color: str = DEFAULT_COLOR,
        figsize: tuple = DEFAULT_FIGSIZE,
        rotate_x: int = DEFAULT_ROTATE_X,
    ) -> None:
        """
        Gráfico de barras para uma série temporal agregada.
 
        Parâmetros
        ----------
        series    : pd.Series indexado pelos períodos (ex: "2021-03", "2022-Q1")
        title     : Título do gráfico
        xlabel    : Rótulo do eixo X
        ylabel    : Rótulo do eixo Y
        color     : Cor das barras (hex ou nome matplotlib)
        figsize   : Tamanho da figura (largura, altura)
        rotate_x  : Rotação dos rótulos do eixo X em graus
 
        Retorna
        -------
        None
        """
        fig, ax = plt.subplots(figsize=figsize)
 
        ax.bar(
            series.index.astype(str),
            series.values,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
 
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        ax.tick_params(axis="x", rotation=rotate_x, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
 
        sns.despine(ax=ax, left=False)
        plt.tight_layout()
        plt.show()