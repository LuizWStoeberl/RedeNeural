let quantidadePorLinha = 1;  
let linhaCount = 0;  


function definirQuantidade() {
    quantidadePorLinha = parseInt(document.getElementById("quantidade").value);
    linhaCount = 0;  
    document.getElementById("btnNovaLinha").disabled = false;  
    document.getElementById("informacoes").innerHTML = '';  
}


function adicionarLinha() {
    linhaCount++;  
    const container = document.getElementById("informacoes");

    const novaLinha = document.createElement("div");
    novaLinha.classList.add("linha");

    
    for (let i = 1; i <= quantidadePorLinha; i++) {
        const idColor = `linha${linhaCount}_cor${i}`;
        const idHexa = `linha${linhaCount}_hexa${i}`;

        novaLinha.innerHTML += `
            <label for="${idColor}">Cor ${i}:</label>
            <input type="color" id="${idColor}" value="#000000">
            <input type="text" id="${idHexa}" readonly>
        `;
    }

    
    novaLinha.innerHTML += `
        <label for="classe${linhaCount}">Classe:</label>
        <input type="text" id="classe${linhaCount}" class="campo-classe" value="Classe ${linhaCount}">
    `;

    container.appendChild(novaLinha);

    
    for (let i = 1; i <= quantidadePorLinha; i++) {
        const corInput = document.getElementById(`linha${linhaCount}_cor${i}`);
        const hexaInput = document.getElementById(`linha${linhaCount}_hexa${i}`);
        hexaInput.value = corInput.value;

        corInput.addEventListener("input", () => {
            hexaInput.value = corInput.value;
        });
    }
}

function enviarDados() {
    const cores = [];
    const labels = [];

    document.querySelectorAll(".linha").forEach((linhaDiv) => {
        const inputsCor = linhaDiv.querySelectorAll('input[type="color"]');
        const linhaCores = Array.from(inputsCor).map(input => input.value);
        cores.push(linhaCores);

        
        const campoClasse = linhaDiv.querySelector('.campo-classe');
        if (campoClasse) {
            labels.push(campoClasse.value || "");
        } else {
            labels.push("");
        }
    });

    const dados = {
        linhaCount: cores.length,
        quantidadePorLinha: cores[0].length,
        cores: cores,
        labels: labels
    };

    fetch('/salvar', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(dados)
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
    });
}