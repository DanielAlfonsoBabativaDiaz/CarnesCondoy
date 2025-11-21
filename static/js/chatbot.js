document.getElementById("chatbot-send").addEventListener("click", sendMessage);
document.getElementById("chatbot-input").addEventListener("keypress", function (e) {
    if (e.key === "Enter") { sendMessage(); }
});

function sendMessage() {
    const input = document.getElementById("chatbot-input");
    const message = input.value.trim();
    if (message === "") return;

    addMessage("user", message);

    fetch("/chatbot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
    })
        .then(res => res.json())
        .then(data => {
            addMessage("bot", data.response);
        });

    input.value = "";
}

function addMessage(sender, text) {
    const box = document.getElementById("chatbot-messages");
    const msg = document.createElement("div");
    msg.classList.add("msg", sender);
    msg.textContent = text;
    box.appendChild(msg);
    box.scrollTop = box.scrollHeight;
}
