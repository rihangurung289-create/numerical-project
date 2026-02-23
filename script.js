// Basic client-side behavior for the Physics & Chemistry problem-solving app.
// This file intentionally contains only simple logic and placeholders.

// Cached state for the currently selected subject.
let currentSubject = "physics";

function setCurrentYear() {
    const yearElement = document.getElementById("year");
    if (yearElement) {
        yearElement.textContent = new Date().getFullYear().toString();
    }
}

// Handle switching between Physics and Chemistry tabs.
function setupSubjectToggle() {
    const toggleContainer = document.getElementById("subjectToggle");
    if (!toggleContainer) return;

    toggleContainer.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) return;

        const subject = target.getAttribute("data-subject");
        if (!subject) return;

        currentSubject = subject;

        // Update button styles so the active subject is clearly highlighted.
        const buttons = toggleContainer.querySelectorAll(".toggle-button");
        buttons.forEach((btn) => {
            if (!(btn instanceof HTMLElement)) return;
            const btnSubject = btn.getAttribute("data-subject");
            btn.classList.toggle("active", btnSubject === currentSubject);
        });
    });
}

// Update the result area with a simple status message.
// Optionally accepts solution steps and a final answer to render.
function showResultMessage(message, { type = "info", steps, finalAnswer } = {}) {
    const resultElement = document.getElementById("result");
    if (!resultElement) return;

    resultElement.innerHTML = "";

    const meta = document.createElement("div");
    meta.className = "result-meta";
    meta.textContent = `Subject: ${currentSubject}`;

    const text = document.createElement("p");
    if (type === "error") {
        text.className = "error-text";
    } else if (type === "success") {
        text.className = "success-text";
    } else {
        text.className = "placeholder-text";
    }
    text.textContent = message;

    resultElement.appendChild(meta);
    resultElement.appendChild(text);

    if (Array.isArray(steps) && steps.length > 0) {
        const listTitle = document.createElement("p");
        listTitle.className = "placeholder-text";
        listTitle.textContent = "Steps:";

        const list = document.createElement("ol");
        steps.forEach((step) => {
            const li = document.createElement("li");
            li.textContent = String(step);
            list.appendChild(li);
        });

        resultElement.appendChild(listTitle);
        resultElement.appendChild(list);
    }

    if (typeof finalAnswer === "string" && finalAnswer.trim() !== "") {
        const final = document.createElement("p");
        final.className = "success-text";
        final.textContent = `Final answer: ${finalAnswer}`;
        resultElement.appendChild(final);
    }
}

// For now, send the problem description to a simple Flask endpoint and
// display whatever message it returns. Later this can be replaced with
// real physics/chemistry calculations.
async function submitProblem(event) {
    event.preventDefault();

    const topicSelect = document.getElementById("topic");
    const problemTextArea = document.getElementById("problemText");

    if (!(topicSelect instanceof HTMLSelectElement) || !(problemTextArea instanceof HTMLTextAreaElement)) {
        showResultMessage("Form controls are not available.", { type: "error" });
        return;
    }

    const topic = topicSelect.value.trim();
    const problemText = problemTextArea.value.trim();

    if (!problemText) {
        showResultMessage("Please enter a brief problem description first.", { type: "error" });
        return;
    }

    showResultMessage("Sending problem to server (placeholder request)...", { type: "info" });

    try {
        const response = await fetch("/api/solve", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                subject: currentSubject,
                topic,
                problemText,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        const payload = await response.json();
        // The server returns only a simple message for now.
        const message = payload && typeof payload.message === "string"
            ? payload.message
            : "Received a response from the server.";

        showResultMessage(message, { type: "success" });
    } catch (error) {
        console.error(error);
        showResultMessage("Could not contact the server. Is the Flask app running?", {
            type: "error",
        });
    }
}

// Send a derivative problem to the dedicated /solve endpoint and
// display the steps and final answer returned by the backend.
async function submitDerivative(event) {
    event.preventDefault();

    const expressionInput = document.getElementById("expression");
    if (!(expressionInput instanceof HTMLInputElement)) {
        showResultMessage("Expression input is not available.", { type: "error" });
        return;
    }

    const expression = expressionInput.value.trim();

    if (!expression) {
        showResultMessage("Please enter an expression in x, for example: x**2 + 3*x.", {
            type: "error",
        });
        return;
    }

    showResultMessage("Computing derivative on the server...", { type: "info" });

    try {
        const response = await fetch("/solve", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                expression,
            }),
        });

        const payload = await response.json();

        if (!response.ok || payload.ok === false) {
            const errorMessage =
                (payload && typeof payload.error === "string" && payload.error) ||
                `Server responded with status ${response.status}`;
            showResultMessage(errorMessage, { type: "error" });
            return;
        }

        showResultMessage("Derivative computed successfully.", {
            type: "success",
            steps: payload.steps,
            finalAnswer: payload.final_answer,
        });
    } catch (error) {
        console.error(error);
        showResultMessage("Could not contact the server. Is the Flask app running?", {
            type: "error",
        });
    }
}

// Upload a problem image, let the backend extract text with OCR,
// then pass that text to the same solver used for typed problems.
async function submitImageProblem(event) {
    event.preventDefault();

    const fileInput = document.getElementById("problemImage");
    const topicSelect = document.getElementById("topic");

    if (!(fileInput instanceof HTMLInputElement) || !fileInput.files || fileInput.files.length === 0) {
        showResultMessage("Please choose an image file containing the problem.", {
            type: "error",
        });
        return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    formData.append("subject", currentSubject);

    if (topicSelect instanceof HTMLSelectElement) {
        formData.append("topic", topicSelect.value.trim());
    }

    showResultMessage("Uploading image and extracting text...", { type: "info" });

    try {
        const response = await fetch("/api/solve", {
            method: "POST",
            body: formData,
        });

        const payload = await response.json();

        if (!response.ok || payload.ok === false) {
            const errorMessage =
                (payload && typeof payload.error === "string" && payload.error) ||
                (payload && typeof payload.message === "string" && payload.message) ||
                `Server responded with status ${response.status}`;
            showResultMessage(errorMessage, { type: "error" });
            return;
        }

        const message =
            (payload && typeof payload.message === "string" && payload.message) ||
            "Processed image and attempted to solve the problem.";

        showResultMessage(message, {
            type: "success",
            steps: payload.steps,
            finalAnswer: payload.final_answer,
        });
    } catch (error) {
        console.error(error);
        showResultMessage("Could not contact the server. Is the Flask app running?", {
            type: "error",
        });
    }
}

function setupFormHandling() {
    const problemForm = document.getElementById("problemForm");
    if (problemForm) {
        problemForm.addEventListener("submit", submitProblem);
    }

    const derivativeForm = document.getElementById("derivativeForm");
    if (derivativeForm) {
        derivativeForm.addEventListener("submit", submitDerivative);
    }

    const imageForm = document.getElementById("imageForm");
    if (imageForm) {
        imageForm.addEventListener("submit", submitImageProblem);
    }
}

// Initialize the frontend once the DOM is ready.
document.addEventListener("DOMContentLoaded", () => {
    setCurrentYear();
    setupSubjectToggle();
    setupFormHandling();
});

