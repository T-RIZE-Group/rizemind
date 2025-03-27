import marimo

__generated_with = "0.11.28"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import pandas as pd

    from sklearn.model_selection import train_test_split
    return pd, pl, train_test_split


@app.cell
def _(pl):
    df = pl.read_csv('/home/mik/Projects/rizemind/examples/tabpfn_local_decentralized/pre_train_test.csv').drop('')
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df, train_test_split):
    df_train, df_sample = train_test_split(df, random_state=0, test_size=0.1)
    return df_sample, df_train


@app.cell
def _(df_sample, df_train):
    df_train.write_csv('/home/mik/Projects/rizemind/examples/tabpfn_local_decentralized/train.csv')
    df_sample.write_csv('/home/mik/Projects/rizemind/examples/tabpfn_local_decentralized/sample.csv')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
