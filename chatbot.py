import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import namedtuple

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    pipeline,
    Pipeline
)
import torch
from peft import PeftModel
import firebase_admin
from firebase_admin import (
    credentials
)
from google.cloud import firestore

MessagePair = namedtuple("paired", "fromUserID content")


class Constants:
    BOT_ID: str = "OIqNXQpz4HQqoFPiPUD01MiJVAu1"
    USERS: str = "users"
    INCOMING_MESSAGES: str = "incomingChats"
    INCOMING_FRIEND_REQUESTS: str = "friendRequests"
    FRIENDS: str = "friends"
    CHATS_INCLUDING: str = "chatsIncludingUser"

    CHATS: str = "chats"
    CHAT_LOGS: str = "chatLogs"

    TEXT: str = "text"
    TIME: str = "time"


@dataclass
class Message:
    chatID: str
    content: str
    contentType: str
    fromUserID: str
    readBy: list
    time: datetime

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return self.__dict__


class Chat:
    chatID: str
    userIDArray: list = []

    def __init__(self, **kwargs):
        self.chatID = kwargs.get("chatID")
        self.userIDArray = kwargs.get("userIDArray")


class ChatRef:
    chatID: str = ""
    notifications: bool = False

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_dict(self):
        return self.__dict__


class friendRef:
    friendID: str
    notifications: bool

    def __init__(self, **kwargs):
        self.friendID = kwargs.get("friendID")
        self.notifications = kwargs.get("notifications")


@dataclass
class Config:
    history_limit: int = 20
    temperature: float = 0.5
    repetition_penalty: float = 1.15
    custom_stop_tokens: str = "<|eot_id|>"
    max_new_tokens: int = 100
    base_model_id: str = "alpindale/Mistral-7B-v0.2-hf"
    lora_weights_path: str = "Yumeng-Liu/MengBotV2"


class ChatBot:
    _base_model: AutoModelForCausalLM
    _eval_tokenizer: AutoTokenizer
    _ft_model: PeftModel
    _generator: Pipeline

    _db: firestore.Client
    _user_ref: firestore.DocumentReference
    _incoming_messages_ref: firestore.CollectionReference
    _incoming_friends_ref: firestore.CollectionReference
    _included_chats_ref: firestore.CollectionReference

    _incoming_message_listener: firestore.Watch
    _incoming_friends_listener: firestore.Watch

    _history: dict

    def __init__(self):
        ChatBot._db = firestore.Client()

        ChatBot._user_ref = ChatBot._db.collection(Constants.USERS).document(Constants.BOT_ID)
        ChatBot._incoming_messages_ref = ChatBot._user_ref.collection(Constants.INCOMING_MESSAGES)
        ChatBot._incoming_friends_ref = ChatBot._user_ref.collection(Constants.INCOMING_FRIEND_REQUESTS)
        ChatBot._included_chats_ref = ChatBot._user_ref.collection(Constants.CHATS_INCLUDING)

        ChatBot._history = {}
        ChatBot._loadHistory()

        # Loads the model
        ChatBot._loadModel()

        # Initialize listener
        ChatBot._incoming_message_listener = ChatBot._incoming_messages_ref.on_snapshot(self._onInboxChange)
        ChatBot._incoming_friends_listener = ChatBot._incoming_friends_ref.on_snapshot(self._onFriendChange)

    @staticmethod
    def _loadModel():
        """
        Loads pre-trained model
        """
        base_model_id = Config.base_model_id
        lora_weights = Config.lora_weights_path

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        ChatBot._base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=base_model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=True
        )

        ChatBot._eval_tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            add_bos_token=True,
            trust_remote_code=True,
            use_fast=True
        )

        # unk token was used as pad token during finetuning, must set the same here
        ChatBot._eval_tokenizer.pad_token = ChatBot._eval_tokenizer.unk_token
        ChatBot._ft_model = PeftModel.from_pretrained(ChatBot._base_model, lora_weights)

        device = torch.device("cuda")
        ChatBot._ft_model.to(device)
        ChatBot._ft_model.eval()

        # end load model
        ChatBot._generator = pipeline(
            "text-generation",
            model=ChatBot._ft_model,
            tokenizer=ChatBot._eval_tokenizer
        )
        print("Model loaded")

    @staticmethod
    def _loadHistory(limit: int = Config.history_limit):
        chat_refs = (ChatBot._included_chats_ref.stream())
        for chat_ref in chat_refs:
            if not chat_ref.exists:
                continue

            chat_obj = chat_ref.to_dict()
            chat_obj = Chat(**chat_obj)

            chatlog_ref = (
                ChatBot._db.collection(Constants.CHATS)
                .document(chat_obj.chatID)
                .collection(Constants.CHAT_LOGS)
            )
            ChatBot._history[chat_obj.chatID] = ChatBot._loadChatLog(chatlog_ref, limit)

    @staticmethod
    def _loadChatLog(
            chat_log_ref: firestore.CollectionReference,
            limit: int = Config.history_limit
    ) -> list:
        query = chat_log_ref.order_by(Constants.TIME, direction=firestore.Query.DESCENDING).limit(limit)
        # query.order_by(Constants.TIME, direction=firestore.Query.ASCENDING)
        logs = [Message(**doc.to_dict()) for doc in query.stream() if doc.exists]
        logs.reverse()
        print("History found:")
        for i in logs:
            print("\t" + i.content)
        return logs

    @staticmethod
    def _generate_with_model(
            eval_prompt,
            temperature,
            repetition_penalty,
            max_new_tokens
    ):
        output = ChatBot._generator(
            eval_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            return_full_text=False  # Ensures only new tokens are returned
        )
        # Extract the generated text
        text_output = output[0]["generated_text"]
        return text_output

    @staticmethod
    def _update_chat_log(message: Message) -> str:
        """
        Updates the chat log on firebase
        :param message: the new message
        :return: the id of the message
        """
        chat_id = message.chatID
        chat_log_ref: firestore.CollectionReference = (
            ChatBot
            ._db.collection(Constants.CHATS)
            .document(chat_id)
            .collection(Constants.CHAT_LOGS)
        )
        message_ref: firestore.DocumentReference = chat_log_ref.add(message.to_dict())[1]
        print(f"Added to chat log at {message_ref.id}")
        return message_ref.id

    @staticmethod
    def _append_to_history(message: Message):
        chat_id = message.chatID

        if chat_id not in ChatBot._history:
            # Add chat to chatsIncludingUser
            ChatBot._history[chat_id] = [message]

            # update cloud
            chat_ref = ChatBot._user_ref.collection(Constants.CHATS_INCLUDING).document(chat_id)
            chat_ref.set({"chatID": chat_id, "notifications": True})

        else:
            ChatBot._history[chat_id] += [message]
            if len(ChatBot._history[chat_id]) > Config.history_limit:
                ChatBot._history[chat_id].pop(0)

    @staticmethod
    def _send_to_chat_members(chat_id: str, message: Message, message_id: str):
        # Send to other chat members, ignoring self
        doc = ChatBot._db.collection(Constants.CHATS).document(chat_id).get()
        if not doc.exists:
            return

        members = Chat(**doc.to_dict()).userIDArray
        for member in members:
            if member == Constants.BOT_ID:
                continue

            inbox_ref = ChatBot._db.collection(Constants.USERS).document(member).collection(Constants.INCOMING_MESSAGES)
            inbox_ref.document(message_id).set(message.to_dict())

    @staticmethod
    def _convert_input(chatID: str) -> str:
        # Append to template
        template = ""
        if chatID not in ChatBot._history:
            raise KeyError(f"chatID: {chatID} not found in history")

        chat_logs: [Message] = ChatBot._history[chatID]
        for message in chat_logs:
            role = "system" if message.fromUserID == Constants.BOT_ID else "user"
            template += f"<start_header_id>{role}<end_header_id>{message.content}{Config.custom_stop_tokens}\n"

        template += "<start_header_id>system<end_header_id>"
        return template

    @staticmethod
    def _extract_output(response: str) -> str:
        # Output extraction
        output = response.split("<start_header_id>user<end_header_id>")[0]
        output = output.split("<|eot_id|>")[0].strip()
        output = output.split("<eot_id>")[0].strip()
        output = output.split("<start_header_id>system<end_header_id>")
        output = [x.strip() for x in output if x and '/' not in x]
        output = "\n".join(output)
        return output

    @staticmethod
    def _generate(chat_id: str) -> [Message]:
        """
        Generate a response with chat_id
        :param chat_id: the chat_id to generate a reply for
        :return: the generated response
        """
        print("Generating response...")

        collected_prompt = ChatBot._convert_input(chat_id)
        output = ChatBot._generate_with_model(
            eval_prompt=collected_prompt,
            temperature=Config.temperature,
            repetition_penalty=Config.repetition_penalty,
            max_new_tokens=Config.max_new_tokens
        )
        output = ChatBot._extract_output(output)
        print(output)
        output = [Message(
            **{
                "chatID": chat_id,
                "fromUserID": Constants.BOT_ID,
                "content": part,
                "contentType": "text",
                "time": datetime.now(),
                "readBy": []
            }
        ) for part in output.split('\n')]
        return output

    @staticmethod
    def _addFriend(new_friend):
        ...

    @staticmethod
    def _onFriendChange(snapshot, changes, read_time):
        for change in changes:
            match change.type.name:
                case "ADDED":
                    new_friend = change.document.to_dict()
                    new_friend = friendRef(**new_friend)
                    ChatBot._addFriend(new_friend)

    @staticmethod
    def _onInboxChange(snapshot, changes, read_time):
        for change in changes:
            match change.type.name:
                case "ADDED":
                    dic = change.document.to_dict()
                    incoming_msg_id = change.document.id
                    incoming_message = Message(**dic)
                    print(f"Received new message:\n {incoming_message.content}")
                    # Remove message from inbox
                    ChatBot._user_ref.collection(Constants.INCOMING_MESSAGES).document(incoming_msg_id).delete()

                    # Update history for input
                    ChatBot._append_to_history(incoming_message)

                    # Generate response
                    ChatBot._generate_response_at(incoming_message.chatID)

    @staticmethod
    def _generate_response_at(chat_id: str):
        # Generate response
        response_msgs = ChatBot._generate(chat_id)
        for response_msg in response_msgs:
            # Update history to include response
            ChatBot._append_to_history(response_msg)

            # Update chat log
            message_id = ChatBot._update_chat_log(response_msg)

            # Sends message
            ChatBot._send_to_chat_members(chat_id, response_msg, message_id)


def _initApp():
    print("Initializing app")
    cred = credentials.Certificate("sdk-keys.json")
    firebase_admin.initialize_app(cred)
    print("App successfully initialized")


def _run():
    ChatBot()
    while True:
        time.sleep(60)


if __name__ == "__main__":
    _initApp()
    _run()
