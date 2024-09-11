from typing import override

import logging

from langchain_core.output_parsers import ListOutputParser
from langchain_core.exceptions import OutputParserException


_logger = logging.getLogger(__name__)
# _logger.setLevel(logging.DEBUG)


class BasicQueryParser(ListOutputParser):
    """Parser of queries that outputs a list."""

    @override
    def parse(self, text: str) -> list[str]:
        _logger.debug(text)
        res: list[str] = []
        for query in text.splitlines():
            if r"{category}" in query:
                res.append(query[query.find("\"") + 1:query.rfind("\"")])
        if not res:
            raise OutputParserException(f"No queries found in:\n{text}")
        return res


class PythonQueryParser(ListOutputParser):
    """Parser of queries that outputs a list."""

    @override
    def parse(self, text: str) -> list[str]:
        _logger.debug(text)
        res: list[str] = []
        for query in text.splitlines():
            if query.strip().startswith("prompts.append"):
                modified_query = query[query.find("\"") + 1:query.rfind("\"")]
                modified_query = modified_query.replace(
                    "\" + category + \"", r"{category}")
                res.append(modified_query)
        if not res:
            raise OutputParserException(f"No queries found in:\n{text}")
        return res
