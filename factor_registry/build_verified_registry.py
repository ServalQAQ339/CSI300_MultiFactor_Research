"""Create a source-verified, non-destructive factor definition registry."""
import ast
import hashlib
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / 'factor_engine.py'
OUTPUT = Path(__file__).resolve().parent / 'original27_registry_verified_v1.json'
BASE = {
    'OPEN_CLOSE_RATIO': ('opening / close - 1', 'open,close', 'candlestick'),
    'HIGH_LOW_RANGE': ('(high - low) / close', 'high,low,close', 'candlestick'),
    'CLOSE_LOCATION': ('(close - low) / (high - low)', 'close,low,high', 'candlestick'),
}
ROLLING = {
    'MOM': ('close / close.shift(w) - 1', 'close', 'momentum'),
    'MA_RATIO': ('close / rolling_mean(close,w) - 1', 'close', 'moving_average'),
    'VOLATILITY': ('rolling_std(close / close.shift(1) - 1,w,ddof=1)', 'close', 'volatility'),
    'VOLUME_RATIO': ('volume / rolling_mean(volume,w)', 'volume', 'volume'),
    'HIGH_POSITION': ('close / rolling_max(high,w) - 1', 'close,high', 'price_position'),
    'LOW_POSITION': ('close / rolling_min(low,w) - 1', 'close,low', 'price_position'),
}


def build_verified_registry(source: Path = SOURCE) -> dict:
    """Description: Verify the expected factor implementation against the source AST.

    Parameters:
        source (Path): Python source file implementing the original factors.
    Returns:
        dict: Versioned definitions and SHA-256 provenance for all 27 factors.
    Notes:
        Source verification is strict to the supplied implementation; it does not certify upstream adjusted prices or point-in-time data.
    """
    content = source.read_text(encoding='utf-8')
    tree = ast.parse(content)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate_stock_factors')
    lines = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Assign)]
    expected = ['opening.div(close).sub(1)', 'high.sub(low).div(close)', 'close.sub(low).div(high.sub(low).replace(0, np.nan))', 'close.div(close.shift(window)).sub(1)', 'close.div(close.rolling(window, min_periods=window).mean()).sub(1)', 'daily_return.rolling(window, min_periods=window).std(ddof=1)', 'volume.div(volume.rolling(window, min_periods=window).mean().replace(0, np.nan))', 'close.div(high.rolling(window, min_periods=window).max()).sub(1)', 'close.div(low.rolling(window, min_periods=window).min()).sub(1)']
    for expression in expected:
        if not any(expression in line for line in lines):
            raise ValueError(f'Unverified expression: {expression}')
    factors = []
    for name, (formula, required, family) in BASE.items():
        factors.append({'factor_id': f'original27:{name}:v1', 'name': name, 'family': family, 'window': None, 'formula': formula, 'requires': required.split(','), 'formula_verified': True})
    for w in (5, 10, 20, 60):
        for prefix, (formula, required, family) in ROLLING.items():
            name = f'{prefix}_{w}'
            factors.append({'factor_id': f'original27:{name}:v1', 'name': name, 'family': family, 'window': w, 'formula': formula.replace('w', str(w)) if False else formula.replace('(w)', f'({w})').replace(',w,', f',{w},'), 'requires': required.split(','), 'formula_verified': True})
    return {'schema_version': 1, 'source_file': 'factors/factor_engine.py', 'source_sha256': hashlib.sha256(content.encode('utf-8')).hexdigest(), 'factor_count': len(factors), 'formula_audit': 'source_verified', 'upstream_adjustment_certified': False, 'historical_membership_pit_certified': False, 'factors': factors}


def main() -> None:
    """Description: Write a separate audited registry without overwriting existing registry files.

    Parameters:
        None: Uses the local source snapshot.
    Returns:
        None: Creates a verified JSON registry.
    Notes:
        Raises if the source no longer matches the audited implementation.
    """
    result = build_verified_registry()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Verified {result["factor_count"]} factors: {OUTPUT}')


if __name__ == '__main__':
    main()
