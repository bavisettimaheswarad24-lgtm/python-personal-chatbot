/* ==========================================================================
   MAHESH AI - Fast Client Interaction Script
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    const introScreen = document.getElementById("intro-screen");
    const skipIntroBtn = document.getElementById("skip-intro-btn");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const messagesViewport = document.getElementById("messages-viewport");
    const messagesList = document.getElementById("messages-list");
    const landingHero = document.getElementById("landing-hero");
    const typingIndicator = document.getElementById("typing-indicator");
    const newChatBtn = document.getElementById("new-chat-btn");
    const promptCards = document.querySelectorAll(".prompt-card");
    const themeToggleBtn = document.getElementById("theme-toggle");
    const mobileMenuBtn = document.getElementById("mobile-menu-btn");
    const sidebar = document.getElementById("sidebar");
    const quickLinksList = document.getElementById("quick-links-list");

    let lastQuery = "";

    // 1. Fast 2s Intro Screen
    const dismissIntro = () => {
        if (!introScreen.classList.contains("hide")) {
            introScreen.classList.add("hide");
            userInput.focus();
        }
    };

    const introTimer = setTimeout(dismissIntro, 2000);
    skipIntroBtn.addEventListener("click", () => {
        clearTimeout(introTimer);
        dismissIntro();
    });
    window.addEventListener("keydown", (e) => {
        if (e.code === "Space" && !introScreen.classList.contains("hide")) {
            clearTimeout(introTimer);
            dismissIntro();
        }
    });

    // 2. Theme Switching
    const savedTheme = localStorage.getItem("mahesh_theme") || "dark";
    document.documentElement.setAttribute("data-theme", savedTheme);

    themeToggleBtn.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme");
        const next = current === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("mahesh_theme", next);
    });

    // 3. Mobile Sidebar Drawer
    mobileMenuBtn.addEventListener("click", () => sidebar.classList.toggle("open"));
    document.addEventListener("click", (e) => {
        if (window.innerWidth <= 850 && !sidebar.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
            sidebar.classList.remove("open");
        }
    });

    // 4. Auto-expand textarea
    userInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 140) + "px";
    });

    // 5. Send on Enter (Shift+Enter for new line)
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // 6. Starter Python Question Cards
    promptCards.forEach(card => {
        card.addEventListener("click", () => {
            const prompt = card.getAttribute("data-prompt");
            userInput.value = prompt;
            chatForm.dispatchEvent(new Event("submit"));
        });
    });

    // 7. Event Delegation for Sidebar Quick Links
    if (quickLinksList) {
        quickLinksList.addEventListener("click", (e) => {
            const item = e.target.closest(".history-item");
            if (item) {
                userInput.value = item.getAttribute("data-q");
                chatForm.dispatchEvent(new Event("submit"));
                if (window.innerWidth <= 850) sidebar.classList.remove("open");
            }
        });
    }

    // 8. New Chat
    newChatBtn.addEventListener("click", () => {
        messagesList.innerHTML = "";
        landingHero.style.display = "flex";
        userInput.value = "";
        userInput.style.height = "auto";
        userInput.focus();
    });

    // 9. Fast Chat Submission
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        lastQuery = text;
        if (landingHero.style.display !== "none") {
            landingHero.style.display = "none";
        }

        renderMessage(text, "user");
        userInput.value = "";
        userInput.style.height = "auto";

        typingIndicator.style.display = "flex";
        messagesViewport.scrollTo({ top: messagesViewport.scrollHeight, behavior: 'smooth' });

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            typingIndicator.style.display = "none";
            renderMessage(data.reply || "No response received.", "bot");
        } catch (err) {
            typingIndicator.style.display = "none";
            renderMessage("⚠️ Connection error. Make sure `python app.py` is running.", "bot");
        }
    });

    function renderMessage(content, sender) {
        const row = document.createElement("div");
        row.className = `message-row ${sender}-row`;

        const avatar = document.createElement("div");
        avatar.className = `message-avatar ${sender}-avatar`;
        avatar.textContent = sender === "bot" ? "M" : "YOU";

        const bubbleWrap = document.createElement("div");
        bubbleWrap.style.display = "flex";
        bubbleWrap.style.flexDirection = "column";

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";

        if (sender === "bot") {
            bubble.innerHTML = marked.parse(content);
            bubble.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));

            const actions = document.createElement("div");
            actions.className = "message-actions";

            const copyBtn = document.createElement("button");
            copyBtn.className = "action-pill-btn";
            copyBtn.textContent = "⎘ Copy Answer";
            copyBtn.onclick = () => {
                navigator.clipboard.writeText(content);
                copyBtn.textContent = "✓ Copied";
                setTimeout(() => copyBtn.textContent = "⎘ Copy Answer", 1800);
            };

            const retryBtn = document.createElement("button");
            retryBtn.className = "action-pill-btn";
            retryBtn.textContent = "↻ Re-ask";
            retryBtn.onclick = () => {
                if (lastQuery) {
                    userInput.value = lastQuery;
                    chatForm.dispatchEvent(new Event("submit"));
                }
            };

            actions.appendChild(copyBtn);
            actions.appendChild(retryBtn);
            bubbleWrap.appendChild(bubble);
            bubbleWrap.appendChild(actions);
        } else {
            bubble.textContent = content;
            bubbleWrap.appendChild(bubble);
        }

        if (sender === "bot") {
            row.appendChild(avatar);
            row.appendChild(bubbleWrap);
        } else {
            row.appendChild(bubbleWrap);
            row.appendChild(avatar);
        }

        messagesList.appendChild(row);
        messagesViewport.scrollTo({ top: messagesViewport.scrollHeight, behavior: 'smooth' });
    }
});
