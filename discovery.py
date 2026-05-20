import pandas as pd
import numpy as np
from graphics import Plots

class DataDiscovery:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.gfx = Plots()


    def verify_null(self) -> str:
        """
        Verifica a presença de valores nulos e nulos implícitos (soft nulls) em um DataFrame.

        Percorre todas as colunas do DataFrame em busca de valores que representam
        ausência de dado de forma textual, como strings vazias, variações de 'N/A',
        'null', 'None', 'nan', entre outros.

        Parâmetros
        ----------
        df : pd.DataFrame
            O DataFrame a ser verificado.

        Retorna
        -------
        str
            Um relatório textual listando, para cada coluna afetada, a quantidade
            de soft nulls encontrados e os valores distintos detectados.
            Caso nenhum soft null seja encontrado, retorna a mensagem
            'Nenhum soft null encontrado.'

        Exemplos
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({'email': ['user@email.com', 'N/A', ''], 'idade': [25, 'none', 30]})
        >>> print(verify_soft_null(df))
        Coluna 'email': 2 soft null(s) encontrado(s) → ['N/A', '']
        Coluna 'idade': 1 soft null(s) encontrado(s) → ['none']
        """

        soft_nulls = ["", "N/A", "NA", "n/a", "na", "null", "NULL", "None", "none", "NaN", "nan", "-", " "]
        
        results = []
        for col in self.df.columns:
            mask = (self.df[col].astype(str).isin(soft_nulls)) | (self.df[col].isnull())

            count = mask.sum()
            if count > 0:
                results.append(f"Column '{col}': {count} soft null(s) found → {self.df[col][mask].unique().tolist()}")
        
        return "\n".join(results) if results else "No soft nulls found."
    
    def continuos_analytics(self, numeric_cols: list) -> str:
        """
        Realiza uma análise descritiva de colunas numéricas contínuas em um DataFrame.

        Para cada coluna numérica contínua, calcula e apresenta as seguintes estatísticas:
        - Média
        - Mediana
        - Desvio padrão
        - Mínimo
        - Máximo
        - Quartis (25%, 50%, 75%)

        Parâmetros
        ----------
        df : pd.DataFrame
            O DataFrame a ser analisado.
        Retorna
        -------
        str
            Um relatório textual listando, para cada coluna numérica contínua, as estatísticas calculadas.
            Caso nenhuma coluna numérica contínua seja encontrada, retorna a mensagem
            'Nenhuma coluna numérica contínua encontrada.' 
    """    
        if len(numeric_cols) == 0:
            return "No continuous numeric columns found."
        results = []
        for col in numeric_cols:
            stats = self.df[col].describe()
            percentiles = np.percentile(self.df[self.df[col].notnull()][col], [1, 5, 10, 20, 25, 50, 75, 80, 90, 95, 99], axis=0)
            results.append(
                f"Column '{col}':\n"
                f"  Mean: {stats['mean']}\n"
                f"  Median: {self.df[col].median()}\n"
                f"  Std Dev: {stats['std']}\n"
                f"  Min: {stats['min']}\n"
                f"  1%: {percentiles[0]}\n"
                f"  5%: {percentiles[1]}\n"
                f"  10%: {percentiles[2]}\n"
                f"  20%: {percentiles[3]}\n"
                f"  25%: {percentiles[4]}\n"
                f"  50%: {percentiles[5]}\n"
                f"  75%: {percentiles[6]}\n"
                f"  80%: {percentiles[7]}\n"
                f"  90%: {percentiles[8]}\n"
                f"  95%: {percentiles[9]}\n"
                f"  99%: {percentiles[10]}\n"
                f"  Max: {stats['max']}"
            )
        
        return "\n\n".join(results) 
    
    
    def categorical_analytics(self, categorical_cols: list) -> str:
        """
        Realiza uma análise descritiva de colunas categóricas em um DataFrame.

        Para cada coluna categórica, calcula e apresenta as seguintes estatísticas:
        - Contagem total de valores
        - Número de categorias distintas
        - Frequência de cada categoria (contagem e porcentagem)

        Parâmetros
        ----------
        df : pd.DataFrame
            O DataFrame a ser analisado.
        
        Retorna
        -------
        str
            Um relatório textual listando, para cada coluna categórica, as estatísticas calculadas.
            Caso nenhuma coluna categórica seja encontrada, retorna a mensagem
            'Nenhuma coluna categórica encontrada.' 
    """    
        if len(categorical_cols) == 0:
            return "No categorical columns found."
        
        results = []
        for col in categorical_cols:
            value_counts = self.df[col].value_counts(dropna=False)
            total_count = len(self.df[col])
            distinct_count = self.df[col].nunique(dropna=False)
            
            category_stats = "\n".join(
                [f"    '{cat}': {count} ({(count/total_count)*100:.2f}%)" for cat, count in value_counts.items()]
            )
            
            results.append(
                f"Column '{col}':\n"
                f"  Total Count: {total_count}\n"
                f"  Distinct Categories: {distinct_count}\n"
                f"  Category Frequencies:\n{category_stats}"
            )
        
        return "\n\n".join(results)

    def date_analytics(self, date_cols: list) -> None:
        """
        Realiza análise univariada de colunas de data.
 
        Para cada coluna informada, gera contagens agregadas em quatro
        granularidades temporais — ano, semestre, trimestre e mês — e
        exibe os respectivos gráficos de barras.
 
        Os gráficos são produzidos pela função `plot_time_series`, que é
        desacoplada desta classe e pode ser reutilizada futuramente para
        outras métricas (média, mediana, soma) com qualquer variável
        contínua agrupada por data.
 
        Parâmetros
        ----------
        date_cols : list
            Lista com os nomes das colunas de data a serem analisadas.
            As colunas devem ser do tipo datetime ou conversíveis via
            pd.to_datetime.
 
        Retorna
        -------
        None
            Exibe os gráficos diretamente. Não retorna objetos.
 
        Exemplo
        -------
        >>> analyzer = DataAnalyzer(df)
        >>> analyzer.date_analytics(["data_contrato", "data_pagamento"])
        """
        granularidades = {
            "Ano": {
                "extractor": lambda s: s.dt.year.astype(str),
                "xlabel": "Ano",
            },
            "Ano-Semestre": {
                "extractor": lambda s: s.dt.year.astype(str)
                    + "-S"
                    + s.dt.quarter.map({1: "1", 2: "1", 3: "2", 4: "2"}),
                "xlabel": "Ano-Semestre",
            },
            "Ano-Trimestre": {
                "extractor": lambda s: s.dt.year.astype(str)
                    + "-Q"
                    + s.dt.quarter.astype(str),
                "xlabel": "Ano-Trimestre",
            },
            "Ano-Mês": {
                "extractor": lambda s: s.dt.to_period("M").astype(str),
                "xlabel": "Ano-Mês",
            },
        }
 
        for col in date_cols:
            if col not in self.df.columns:
                print(f"[AVISO] Coluna '{col}' não encontrada no DataFrame. Pulando.")
                continue
 
            # Garante tipo datetime
            serie = pd.to_datetime(self.df[col], errors="coerce")
 
            n_nulos = serie.isna().sum()
            if n_nulos > 0:
                print(f"[INFO] '{col}': {n_nulos:,} valores nulos ignorados nas agregações.")
 
            serie = serie.dropna()
 
            print(f"\n{'='*60}")
            print(f"  ANÁLISE DE DATA: {col}")
            print(f"  Registros válidos : {len(serie):,}")
            print(f"  Período           : {serie.min().date()} → {serie.max().date()}")
            print(f"{'='*60}")
 
            for nome_gran, config in granularidades.items():
                periodo = config["extractor"](serie)
                contagem = periodo.value_counts().sort_index()
 
                self.gfx.bar(
                    series=contagem,
                    title=f"{col}  ·  Distribuição por {nome_gran}",
                    xlabel=config["xlabel"],
                    ylabel="Quantidade de registros",
                )
 
 
# =============================================================================
# EXEMPLO DE REUSO — média de uma variável contínua agrupada por mês
# =============================================================================
#
# Se no futuro você quiser plotar a média de `valor_contrato` por mês:
#
#   media_mensal = (
#       df.assign(periodo=pd.to_datetime(df["data_contrato"]).dt.to_period("M").astype(str))
#         .groupby("periodo")["valor_contrato"]
#         .mean()
#         .sort_index()
#   )
#
#   plot_time_series(
#       series=media_mensal,
#       title="Valor médio de contrato por mês",
#       xlabel="Ano-Mês",
#       ylabel="Valor médio (R$)",
#       color="#DD8452",
#   )
#
# =============================================================================