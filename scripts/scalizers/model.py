
def scale_model(model: str, model_list: list[str]) -> float | None:
    if model is None:
        return None
    model = str(model).strip().title()
    if model in model_list:
        index = model_list.index(model)
        return index / (len(model_list) - 1)
    else:
        return None