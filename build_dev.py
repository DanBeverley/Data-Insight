from e2b import Template, default_build_logger
from .template import template


if __name__ == "__main__":
    Template.build(
        template,
        alias="data-insight-sandbox-dev",
        on_build_logs=default_build_logger(),
    )
