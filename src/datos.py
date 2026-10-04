"""Valida y prepara los datos de la competencia para el equipo de ML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "salida"

TARGET = "objetivo"
IDENTIFIER = "id_cliente"
MONTH = "mes"
MISSING_CATEGORY = "__faltante__"
UNKNOWN_CATEGORY = "__desconocido__"

NUMERIC_COLUMNS = [
    "edad",
    "ingresos",
    "ratio_deuda_ingresos",
    "antiguedad_cuenta_meses",
    "numero_productos",
    "saldo_promedio",
    "dias_ultima_transaccion",
    "antiguedad_direccion_meses",
    "visitas_web_ultimos_90_dias",
    "distancia_sucursal_km",
    "dia_preferido_pago",
    "dias_ultima_interaccion",
]
CATEGORICAL_COLUMNS = [
    "ocupacion",
    "region",
    "canal_adquisicion",
    "banda_riesgo",
    "dispositivo_principal",
]
BOOLEAN_COLUMNS = [
    "tiene_tarjeta_credito",
    "activo_movil",
    "es_nuevo_cliente",
    "tiene_prestamo",
    "tiene_seguro",
]
BASE_FEATURE_COLUMNS = [
    MONTH,
    *NUMERIC_COLUMNS,
    *BOOLEAN_COLUMNS,
    *CATEGORICAL_COLUMNS,
]
TRAIN_COLUMNS = [IDENTIFIER, *BASE_FEATURE_COLUMNS, TARGET]
TEST_COLUMNS = [IDENTIFIER, *BASE_FEATURE_COLUMNS]


class DataValidationError(ValueError):
    """Error de esquema o calidad que impide exportar datos fiables."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {path}")
    return pd.read_csv(path, low_memory=False)


def _check_columns(frame: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise DataValidationError(
            f"Esquema inválido en {name}. Columnas faltantes: {missing}; "
            f"columnas inesperadas: {extra}."
        )


def _parse_boolean(series: pd.Series, column: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.casefold()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "si": True,
        "sí": True,
    }
    parsed = normalized.map(mapping).astype("boolean")
    invalid = series.notna() & parsed.isna()
    if invalid.any():
        values = series[invalid].astype(str).drop_duplicates().head(5).tolist()
        raise DataValidationError(
            f"Valores booleanos inválidos en {column}: {values}."
        )
    return parsed


def _normalize_frame(
    frame: pd.DataFrame, expected: list[str], name: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    _check_columns(frame, expected, name)
    result = frame.loc[:, expected].copy()
    parse_errors: dict[str, int] = {}

    for column in [IDENTIFIER, MONTH, *NUMERIC_COLUMNS]:
        converted = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].notna() & converted.isna()
        if invalid.any():
            parse_errors[column] = int(invalid.sum())
        result[column] = converted

    for column in BOOLEAN_COLUMNS:
        result[column] = _parse_boolean(result[column], column)

    for column in CATEGORICAL_COLUMNS:
        values = result[column].astype("string").str.strip().str.casefold()
        result[column] = values.mask(values.eq(""), pd.NA)

    if name == "train":
        converted_target = pd.to_numeric(result[TARGET], errors="coerce")
        invalid = result[TARGET].notna() & converted_target.isna()
        if invalid.any():
            parse_errors[TARGET] = int(invalid.sum())
        result[TARGET] = converted_target

    if parse_errors:
        raise DataValidationError(
            f"No se pudieron convertir valores numéricos en {name}: {parse_errors}."
        )
    return result, parse_errors


def _validate_data(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> dict[str, Any]:
    issues: list[str] = []

    for name, frame in (("train", train), ("test", test)):
        nulls = frame.isna().sum()
        present_nulls = {column: int(count) for column, count in nulls.items() if count}
        required_columns = [IDENTIFIER, MONTH]
        if name == "train":
            required_columns.append(TARGET)
        required_nulls = {
            column: present_nulls[column]
            for column in required_columns
            if column in present_nulls
        }
        if required_nulls:
            issues.append(f"{name}: faltan valores en columnas clave: {required_nulls}")

        numeric = frame.select_dtypes(include="number")
        non_finite = {
            column: int((numeric[column].notna() & ~np.isfinite(numeric[column])).sum())
            for column in numeric
            if (numeric[column].notna() & ~np.isfinite(numeric[column])).any()
        }
        if non_finite:
            issues.append(f"{name}: hay valores infinitos: {non_finite}")

        if frame[IDENTIFIER].isna().any() or (frame[IDENTIFIER] <= 0).any():
            issues.append(f"{name}: id_cliente debe ser un entero positivo.")
        elif not np.equal(frame[IDENTIFIER], np.floor(frame[IDENTIFIER])).all():
            issues.append(f"{name}: id_cliente debe ser un entero.")

        month_values = frame[MONTH]
        month_parts = month_values.map(
            lambda value: (
                str(int(value))
                if pd.notna(value) and float(value).is_integer()
                else ""
            )
        )
        invalid_months = ~month_parts.str.fullmatch(r"\d{4}(0[1-9]|1[0-2])")
        if invalid_months.any():
            issues.append(f"{name}: mes debe respetar el formato AAAAMM.")

        duplicate_keys = int(frame.duplicated([IDENTIFIER, MONTH]).sum())
        if duplicate_keys:
            issues.append(
                f"{name}: hay {duplicate_keys} claves duplicadas (id_cliente, mes)."
            )

        domain_checks = {
            "edad": frame["edad"].lt(0),
            "ingresos": frame["ingresos"].lt(0),
            "ratio_deuda_ingresos": ~frame["ratio_deuda_ingresos"].between(0, 1),
            "antiguedad_cuenta_meses": frame["antiguedad_cuenta_meses"].lt(0),
            "numero_productos": frame["numero_productos"].lt(0),
            "dias_ultima_transaccion": frame["dias_ultima_transaccion"].lt(0),
            "antiguedad_direccion_meses": frame["antiguedad_direccion_meses"].lt(0),
            "visitas_web_ultimos_90_dias": frame["visitas_web_ultimos_90_dias"].lt(0),
            "distancia_sucursal_km": frame["distancia_sucursal_km"].lt(0),
            "dia_preferido_pago": ~frame["dia_preferido_pago"].between(1, 31),
            "dias_ultima_interaccion": frame["dias_ultima_interaccion"].lt(0),
        }
        for column, invalid in domain_checks.items():
            if invalid.fillna(False).any():
                issues.append(
                    f"{name}: {column} tiene {int(invalid.sum())} valores fuera "
                    "de rango."
                )

        integer_columns = [
            IDENTIFIER,
            MONTH,
            "edad",
            "antiguedad_cuenta_meses",
            "numero_productos",
            "dias_ultima_transaccion",
            "antiguedad_direccion_meses",
            "visitas_web_ultimos_90_dias",
            "dia_preferido_pago",
            "dias_ultima_interaccion",
        ]
        for column in integer_columns:
            values = frame[column].dropna()
            if not np.equal(values, np.floor(values)).all():
                issues.append(f"{name}: {column} debe contener valores enteros.")

    invalid_target = ~train[TARGET].isin([0, 1])
    if invalid_target.any():
        issues.append(
            f"train: objetivo debe ser 0/1; hay {int(invalid_target.sum())} "
            "valores inválidos."
        )

    if not sample.columns.tolist() == [IDENTIFIER, "prediccion"]:
        issues.append(
            "sample_submission.csv debe contener exactamente "
            "id_cliente,prediccion en ese orden."
        )
    elif len(sample) != len(test):
        issues.append(
            f"El sample submission tiene {len(sample)} filas, pero test tiene "
            f"{len(test)}."
        )
    elif not sample[IDENTIFIER].equals(test[IDENTIFIER]):
        issues.append(
            "El orden o los id_cliente del sample submission no coinciden con test."
        )

    if not issues and train[TARGET].groupby(train[IDENTIFIER]).sum().gt(1).any():
        issues.append(
            "Hay clientes con más de un objetivo positivo; revisar la definición "
            "de primera conversión."
        )

    if issues:
        raise DataValidationError(
            "La validación encontró problemas y no se exportaron datos:\n- "
            + "\n- ".join(issues)
        )

    return {
        "filas_train": len(train),
        "filas_test": len(test),
        "columnas_train": len(train.columns),
        "columnas_test": len(test.columns),
        "nulos_train": {
            column: int(count)
            for column, count in train.isna().sum().items()
            if count
        },
        "nulos_test": {
            column: int(count) for column, count in test.isna().sum().items() if count
        },
        "duplicados_id_mes_train": int(train.duplicated([IDENTIFIER, MONTH]).sum()),
        "duplicados_id_mes_test": int(test.duplicated([IDENTIFIER, MONTH]).sum()),
        "meses_train": sorted(int(value) for value in train[MONTH].unique()),
        "meses_test": sorted(int(value) for value in test[MONTH].unique()),
        "distribucion_objetivo": {
            str(int(key)): int(value)
            for key, value in train[TARGET].value_counts().sort_index().items()
        },
        "tasa_positiva": float(train[TARGET].mean()),
        "clientes_con_mas_de_un_objetivo_positivo": int(
            train.groupby(IDENTIFIER)[TARGET].sum().gt(1).sum()
        ),
        "sample_submission_coincide_con_test": True,
        "notas": [
            "No se eliminaron filas ni se recortaron outliers.",
            "id_cliente se conserva en los CSV limpios y en test_ids.csv, "
            "pero se excluye de las features para evitar fuga por identificador.",
            "mes se conserva como feature temporal.",
            "Los nulos de features se imputan usando únicamente estadísticas de train.",
        ],
    }


def _make_model_matrices(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_features = train.drop(columns=[IDENTIFIER, TARGET]).copy()
    test_features = test.drop(columns=[IDENTIFIER]).copy()
    train_features = train_features.drop(columns=CATEGORICAL_COLUMNS)
    test_features = test_features.drop(columns=CATEGORICAL_COLUMNS)

    feature_contract: dict[str, Any] = {
        "id_cliente_excluido": True,
        "mes_incluido": True,
        "escalado_numerico": False,
        "codificacion_categorica": "one-hot; categorías aprendidas exclusivamente de train",
        "categorias_por_columna": {},
        "imputacion_numerica": {},
    }
    for column in NUMERIC_COLUMNS:
        median = float(train[column].median())
        feature_contract["imputacion_numerica"][column] = median
        train_features[f"{column}__faltante"] = train[column].isna().astype("int8")
        test_features[f"{column}__faltante"] = test[column].isna().astype("int8")
        train_features[column] = train[column].fillna(median)
        test_features[column] = test[column].fillna(median)

    for column in BOOLEAN_COLUMNS:
        train_features[f"{column}__faltante"] = train[column].isna().astype("int8")
        test_features[f"{column}__faltante"] = test[column].isna().astype("int8")
        train_features[column] = train[column].fillna(False).astype("int8")
        test_features[column] = test[column].fillna(False).astype("int8")

    for column in CATEGORICAL_COLUMNS:
        train_values = train[column].fillna(MISSING_CATEGORY).astype(str)
        test_values = test[column].fillna(MISSING_CATEGORY).astype(str)
        categories = sorted(set(train_values.unique()))
        if MISSING_CATEGORY not in categories:
            categories.append(MISSING_CATEGORY)
        if UNKNOWN_CATEGORY not in categories:
            categories.append(UNKNOWN_CATEGORY)
        feature_contract["categorias_por_columna"][column] = categories

        train_values = train_values.where(train_values.isin(categories), UNKNOWN_CATEGORY)
        test_values = test_values.where(test_values.isin(categories), UNKNOWN_CATEGORY)
        train_dummies = pd.get_dummies(train_values, prefix=column, dtype="int8")
        test_dummies = pd.get_dummies(test_values, prefix=column, dtype="int8")
        expected_columns = [f"{column}_{category}" for category in categories]
        train_features = pd.concat(
            [train_features, train_dummies.reindex(columns=expected_columns, fill_value=0)],
            axis=1,
        )
        test_features = pd.concat(
            [test_features, test_dummies.reindex(columns=expected_columns, fill_value=0)],
            axis=1,
        )

    train_features = train_features.astype("float32")
    test_features = test_features.astype("float32")
    if train_features.columns.tolist() != test_features.columns.tolist():
        raise DataValidationError("Las features de train y test no quedaron alineadas.")
    return train_features, test_features, feature_contract


def _impute_clean_frames(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    clean_train = train.copy()
    clean_test = test.copy()
    imputations: dict[str, Any] = {"numeric_medians_from_train": {}, "other": {}}

    for column in NUMERIC_COLUMNS:
        median = float(train[column].median())
        imputations["numeric_medians_from_train"][column] = median
        clean_train[column] = clean_train[column].fillna(median)
        clean_test[column] = clean_test[column].fillna(median)

    for column in BOOLEAN_COLUMNS:
        imputations["other"][column] = "false"
        clean_train[column] = clean_train[column].fillna(False).astype(bool)
        clean_test[column] = clean_test[column].fillna(False).astype(bool)

    for column in CATEGORICAL_COLUMNS:
        imputations["other"][column] = MISSING_CATEGORY
        clean_train[column] = clean_train[column].fillna(MISSING_CATEGORY)
        clean_test[column] = clean_test[column].fillna(MISSING_CATEGORY)

    return clean_train, clean_test, imputations


def prepare_data(
    train_path: Path, test_path: Path, sample_path: Path, output_dir: Path
) -> dict[str, Any]:
    train_raw = _read_csv(train_path)
    test_raw = _read_csv(test_path)
    sample = _read_csv(sample_path)

    train, _ = _normalize_frame(train_raw, TRAIN_COLUMNS, "train")
    test, _ = _normalize_frame(test_raw, TEST_COLUMNS, "test")
    quality_report = _validate_data(train, test, sample)
    train_features, test_features, feature_contract = _make_model_matrices(
        train, test
    )
    clean_train, clean_test, imputations = _impute_clean_frames(train, test)
    quality_report["imputaciones"] = imputations

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_train.to_csv(output_dir / "train_limpio.csv", index=False)
    clean_test.to_csv(output_dir / "test_limpio.csv", index=False)
    train_features.assign(objetivo=train[TARGET].astype("int8")).to_csv(
        output_dir / "train_modelo.csv", index=False
    )
    test_features.to_csv(output_dir / "test_modelo.csv", index=False)
    pd.DataFrame(
        {
            "orden_test": np.arange(len(test), dtype="int32"),
            IDENTIFIER: test[IDENTIFIER].astype("int64"),
        }
    ).to_csv(output_dir / "test_ids.csv", index=False)
    with (output_dir / "reporte_calidad.json").open("w", encoding="utf-8") as file:
        json.dump(quality_report, file, ensure_ascii=False, indent=2)
    with (output_dir / "contrato_features.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(feature_contract, file, ensure_ascii=False, indent=2)

    return {
        **quality_report,
        "features_modelo": train_features.shape[1],
        "directorio_salida": str(output_dir),
        "archivos_generados": [
            "train_limpio.csv",
            "test_limpio.csv",
            "train_modelo.csv",
            "test_modelo.csv",
            "test_ids.csv",
            "reporte_calidad.json",
            "contrato_features.json",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpia, valida y prepara los datos de la competencia para ML."
    )
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test.csv")
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=DATA_DIR / "sample_submission.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    summary = prepare_data(
        args.train, args.test, args.sample_submission, args.output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
