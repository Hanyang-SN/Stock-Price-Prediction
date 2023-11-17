// HTML 요소 생성
const buttonContainer = document.createElement('div');
buttonContainer.style.textAlign = 'center';

// 8개의 버튼 생성
for (let i = 1; i <= 8; i++) {
  const button = document.createElement('button');
  button.textContent = `버튼 ${i}`;
  button.onclick = function() {
    sendData(`Button ${i}`);
  };
  buttonContainer.appendChild(button);
}

// 버튼들을 body에 추가
document.body.appendChild(buttonContainer);

// 사용자가 선택한 버튼 정보를 백엔드로 전송하는 함수
function sendData(buttonName) {
  // 실제로는 fetch API를 사용하여 서버로 데이터를 보냅니다.
  // 서버 주소와 엔드포인트는 실제 환경에 맞게 수정해야 합니다.
  fetch('http://localhost:3000/api/button', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ button: buttonName }) // 버튼 정보를 JSON 형식으로 전송
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(data => {
    // 서버로부터 받은 응답에 대한 처리
    console.log('백엔드로부터의 응답:', data);
    // 필요한 추가 작업을 수행하세요
  })
  .catch(error => {
    console.error('There has been a problem with your fetch operation:', error);
  });
}
