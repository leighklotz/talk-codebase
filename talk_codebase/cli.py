import os
import traceback

import fire
import yaml


from talk_codebase.llm import Session


def get_config():
    home_dir = os.path.expanduser("~")
    config_path = os.path.join(home_dir, ".talk-codebase.yaml")
    print(f"config path {config_path}")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    return config


def save_config(config):
    home_dir = os.path.expanduser("~")
    config_path = os.path.join(home_dir, ".talk-codebase.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def configure():
    config = get_config()
    api_key = input("🤖 Enter your OpenAI API key: ")
    model_name = input("🤖 Enter your model name (default: gpt-3.5-turbo): ") or "gpt-3.5-turbo"
    config["api_key"] = api_key
    config["model_name"] = model_name
    save_config(config)


def loop(session:Session):
    while True:
        question = input("👉 ")
        if not question:
            print("🤖 Please enter a question.")
            continue
        if question.lower() in ('exit', 'quit'):
            break
        result = session.send_question(question)


def chat(root_dir):
    try:
        config = get_config()
        api_key = config.get("api_key")
        model_name = config.get("model_name")
        if not (api_key and model_name):
            configure()
            chat(root_dir)
        session = Session(root_dir, api_key, model_name)
        loop(session)
    except KeyboardInterrupt:
        print("\n🤖 Bye!")
    except Exception as e:
        if str(e) == "<empty message>":
            print("🤖 Please configure your API key.")
            configure()
            chat(root_dir)
        else:
            print(f"\n🤖 Error: {e}")
            traceback.print_exc()


def main():
    fire.Fire({
        "chat": chat,
        "configure": configure,
    })


if __name__ == "__main__":
    main()
