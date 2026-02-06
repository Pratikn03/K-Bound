"""CLI wrapper to validate attention fusion input schema."""

from uais.fusion.attention.validate_fusion_inputs import validate_attention_inputs


def main() -> None:
    validate_attention_inputs()


if __name__ == "__main__":
    main()
