def extract_xml_answer(text: str) -> str:
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()
