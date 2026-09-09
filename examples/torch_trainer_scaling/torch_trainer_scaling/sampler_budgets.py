from __future__ import annotations

MONTE_CARLO_MAX_SAMPLES_BY_TRAINER_COUNT = {
    8: 256,
    9: 512,
    10: 1024,
    11: 2048,
    12: 4096,
    13: 5041,
    14: 4796,
    15: 4619,
    16: 4514,
}

DETERMINISTIC_MAX_SAMPLES_BY_TRAINER_COUNT = {
    8: 256,
    9: 512,
    10: 1024,
    11: 2048,
    12: 3448,
    13: 2461,
    14: 3224,
    15: 2393,
    16: 3139,
}

FIXED_METHOD_SAMPLE_BUDGETS = {
    "antithetic": 3725,
}

SSK_LAST3_MAX_SAMPLES_BY_TRAINER_COUNT = {
    8: 256,
    9: 512,
    10: 1024,
    11: 1825,
    12: 1935,
    13: 1677,
    14: 1728,
    15: 1599,
    16: 1641,
}

STRATIFIED_ANTITHETIC_MAX_EMITTED_SAMPLES_BY_TRAINER_COUNT = {
    8: 256,
    9: 512,
    10: 666,
    11: 900,
    12: 798,
    13: 742,
    14: 674,
    15: 672,
    16: 598,
}

STRATIFIED_ONLY_INTERPOLATED_SAMPLES_BY_TRAINER_COUNT = {
    8: 256,
    9: 512,
    10: 666,
    11: 586,
    12: 528,
    13: 472,
    14: 444,
    15: 418,
    16: 392,
}
STRATIFIED_ONLY_WITH_DUPLICATES_INTERPOLATED_SAMPLES_BY_TRAINER_COUNT = {
    8: 256,
    9: 512,
    10: 1005,
    11: 1046,
    12: 910,
    13: 743,
    14: 618,
    15: 616,
    16: 621,
}

ANTITHETIC_STRATIFIED_INTERPOLATED_SAMPLES_BY_TRAINER_COUNT = {
    8: 256,
    9: 512,
    10: 1024,
    11: 2048,
    12: 2554,
    13: 2532,
    14: 2505,
    15: 2504,
    16: 2474,
}


def _normalize_method_name(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_")
    if normalized == "ssk_last3":
        return "ssk"
    return normalized


def _budget_from_table(
    *,
    table: dict[int, int],
    trainer_count: int,
    description: str,
) -> int:
    try:
        return table[trainer_count]
    except KeyError as exc:
        supported = ",".join(str(count) for count in sorted(table))
        raise ValueError(
            f"Missing hardcoded {description} sample budget for trainer count "
            f"{trainer_count}. Supported counts: {supported}."
        ) from exc


def monte_carlo_budget_for_trainer_count(trainer_count: int) -> int:
    return _budget_from_table(
        table=MONTE_CARLO_MAX_SAMPLES_BY_TRAINER_COUNT,
        trainer_count=trainer_count,
        description="monte-carlo",
    )


def deterministic_budget_for_trainer_count(trainer_count: int) -> int:
    return _budget_from_table(
        table=DETERMINISTIC_MAX_SAMPLES_BY_TRAINER_COUNT,
        trainer_count=trainer_count,
        description="deterministic",
    )


def ssk_last3_budget_for_trainer_count(trainer_count: int) -> int:
    return _budget_from_table(
        table=SSK_LAST3_MAX_SAMPLES_BY_TRAINER_COUNT,
        trainer_count=trainer_count,
        description="ssk-last3",
    )


def stratified_antithetic_budget_for_trainer_count(trainer_count: int) -> int:
    return _budget_from_table(
        table=STRATIFIED_ANTITHETIC_MAX_EMITTED_SAMPLES_BY_TRAINER_COUNT,
        trainer_count=trainer_count,
        description="stratified-antithetic emitted",
    )


def stratified_budget_for_trainer_count(trainer_count: int) -> int:
    return _budget_from_table(
        table=STRATIFIED_ONLY_INTERPOLATED_SAMPLES_BY_TRAINER_COUNT,
        trainer_count=trainer_count,
        description="stratified-only",
    )


def stratified_with_duplicates_budget_for_trainer_count(trainer_count: int) -> int:
    return _budget_from_table(
        table=STRATIFIED_ONLY_WITH_DUPLICATES_INTERPOLATED_SAMPLES_BY_TRAINER_COUNT,
        trainer_count=trainer_count,
        description="stratified-only-with-duplicates",
    )


def antithetic_stratified_budget_for_trainer_count(trainer_count: int) -> int:
    return _budget_from_table(
        table=ANTITHETIC_STRATIFIED_INTERPOLATED_SAMPLES_BY_TRAINER_COUNT,
        trainer_count=trainer_count,
        description="antithetic-stratified",
    )


def configured_budget_for_method(
    *,
    method: str,
    trainer_count: int,
    override_budgets: dict[str, int] | None = None,
    fallback_budget: int,
    ssk_max_missing_trainers: int,
) -> int:
    normalized_method = _normalize_method_name(method)
    override_budgets = override_budgets or {}

    if normalized_method == "exact":
        return 1 << trainer_count
    if normalized_method in override_budgets:
        return override_budgets[normalized_method]
    if normalized_method == "monte_carlo":
        return monte_carlo_budget_for_trainer_count(trainer_count)
    if normalized_method == "deterministic":
        return deterministic_budget_for_trainer_count(trainer_count)
    if normalized_method in FIXED_METHOD_SAMPLE_BUDGETS:
        return FIXED_METHOD_SAMPLE_BUDGETS[normalized_method]
    if normalized_method == "stratified":
        return stratified_budget_for_trainer_count(trainer_count)
    if normalized_method == "stratified_with_duplicates":
        return stratified_with_duplicates_budget_for_trainer_count(trainer_count)
    if normalized_method == "antithetic_stratified":
        return antithetic_stratified_budget_for_trainer_count(trainer_count)
    if normalized_method == "stratified_antithetic":
        return stratified_antithetic_budget_for_trainer_count(trainer_count)
    if normalized_method == "ssk":
        if ssk_max_missing_trainers != 3:
            return fallback_budget
        return ssk_last3_budget_for_trainer_count(trainer_count)
    return fallback_budget
