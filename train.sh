#!/bin/bash
source ./.venv/bin/activate

source ./.env

read -p "Please select model [vitnet=v/unet=u]: " model

if [ "$model" == "v" ]; then
    python3 -m src.models.vitnet.train | tee logs/train-v.log
elif [ "$model" == "u" ]; then
    python3 -m src.models.unet.train | tee logs/train-u.log
else
    echo "Model unknown."
    exit 1
fi

curl --request POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendDocument" --form "chat_id=${CHAT_ID}" --form "document=@logs/train-${model}.log" --form "caption=Finished training"
