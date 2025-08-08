import json

import requests
import torch
import yaml


class SlackSender:
    def __init__(self, slack_webhook, config, exp_name, args):
        """Initializes the SlackSender with the webhook URL."""
        self.webhook_url = slack_webhook
        self.config = config
        self.user_mentioned = self._get_slack_user(exp_name, args)

    def send_slack_message(self, message_blocks):
        """Sends a message to Slack with optional file attachment."""
        try:
            data = json.dumps({"blocks": message_blocks})
            response = requests.post(self.webhook_url, data=data)
            response.raise_for_status()
            print("Slack message sent successfully!")
        except requests.exceptions.RequestException as e:
            print(f"Error sending Slack message: {e}")

    def create_slack_message(self, status, message, training_name, training_env):
        """Creates the Slack message blocks."""
        if self.user_mentioned is not None:
            mention_string = f"<@{self.user_mentioned}>"
        else:
            mention_string = ""

        if status == "start":
            config_str = yaml.dump(self.config, indent=2, default_flow_style=False)
            if len(config_str) > 2000:
                config_str = config_str[-1990:] + " (...) "
            if torch.cuda.is_available():
                message = f"{mention_string} Your training job *{training_name}* has started with *{torch.cuda.device_count()}* GPU(s) *{torch.cuda.get_device_name(0)}*.\n\n```\n{config_str}\n```"
            else:
                message = f"{mention_string} Your training job *{training_name}* has started with *CPU*.\n\n```\n{config_str}\n```"
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":yellow_circle: {message}",
                    },
                },
            ]

        elif status == "error":
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":red_circle: {mention_string} Your training on env *{training_env}* named *{training_name}* has terminated.",
                    },
                },
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Reason: *{message}*"}},
            ]
        elif status == "constants checkpoint":
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":checkpoint: {mention_string} Your training on env *{training_env}* named *{training_name}* has saved the Constants.yaml Checkpoint!",
                    },
                },
                {"type": "section", "text": {"type": "mrkdwn", "text": f"{message}"}},
            ]
        else:  # status == "finished"
            blocks = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f":green_circle: {mention_string} Your training has just finished!"},
                },
                {"type": "section", "text": {"type": "mrkdwn", "text": f"Training Name: *{training_name}*\nEnvironment: *{training_env}*"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"{message}"}},
            ]

        return blocks

    def _get_slack_user(self, exp_name, args):
        if args.slack_users:
            data = json.loads(args.slack_users)
            for item in data:
                if item["user"].upper() in exp_name.upper():
                    return str(item["id"])
        return None
