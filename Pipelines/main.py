import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.compose import make_column_selector
from sklearn.model_selection import GridSearchCV

def filter_data(data):
    data2 = data.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long']
    return data2.drop(columns_to_drop, axis=1)


def calculate_outliers(data):
    data2 = data.copy()
    q25 = data2.quantile(0.25)
    q75 = data2.quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
    return boundaries


def remove_outliers(data):
    data2 = data.copy()
    boundaries = calculate_outliers(data2['year'])
    data2.loc[data2['year'] < boundaries[0], 'year'] = round(boundaries[0])
    data2.loc[data2['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return data2


def short_model(data2):
    # data2 = data.copy()
    if not pd.isna(data2):
        return data2.lower().split(' ')[0]
    else:
        return data2


def predict_model(data):
    data2 = data.copy()
    data2.loc[:, 'short_model'] = data['model'].apply(
        lambda x: x.lower().split(' ')[0] if not pd.isna(x) else x
    )
    return data2


def age_category(data):
    data2 = data.copy()
    data2.loc[:, 'age_category'] = data2['year'].apply(
        lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return data2


def main():
    df = pd.read_csv('homework.csv')
    print('Car price prediction Pipeline')

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor1 = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('remove_outliers', FunctionTransformer(remove_outliers)),
        ('short_model', FunctionTransformer(predict_model)),
        ('age_category', FunctionTransformer(age_category)),
    ])

    preprocessor2= ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])


    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64))
    )

    best_score = .0
    best_pipe = None
    for model in models:
        data = df.copy()
        pipe = Pipeline(steps=[
            ('preprocessor1', preprocessor1),
            ('preprocessor2', preprocessor2),
            ('classifier', model)
        ])
        score = cross_val_score(pipe,
                                data.drop('price_category', axis=1),
                                data['price_category'],
                                cv=4,
                                scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'car_price.pkl')


if __name__ == '__main__':
    main()
