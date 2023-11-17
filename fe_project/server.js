// 프론트엔드 코드
// fetch('http://localhost:3000/api/data')
//   .then(response => {
//     if (!response.ok) {
//       throw new Error('Network response was not ok');
//     }
//     return response.json();
//   })
//   .then(data => {
//     // 받은 데이터를 처리하는 로직
//     console.log(data);
//   })
//   .catch(error => {
//     console.error('There has been a problem with your fetch operation:', error);
//   });





//프론트에서 백으로 get요청을 보내는 코드

fetch('http://ec2-13-124-132-21.ap-northeast-2.compute.amazonaws.com:1235/data')
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(data => {
    // 받은 데이터를 처리하는 로직을 작성합니다.
    console.log(data);
  })
  .catch(error => {
    console.error('There has been a problem with your fetch operation:', error);
  });

//post요청을 보내는 경우

//   fetch('https://your-backend-url.com/api/save', {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//       // 다른 필요한 헤더가 있다면 추가할 수 있습니다.
//     },
//     body: JSON.stringify({ key: 'value' }) // 전송할 데이터 객체를 JSON 형태로 변환하여 보냅니다.
//   })
//     .then(response => {
//       if (!response.ok) {
//         throw new Error('Network response was not ok');
//       }
//       return response.json();
//     })
//     .then(data => {
//       // 성공적으로 데이터를 보낸 후의 처리 로직을 작성합니다.
//       console.log(data);
//     })
//     .catch(error => {
//       console.error('There has been a problem with your fetch operation:', error);
//     });
  


// const express = require('express');
// const app = express();

// app.listen(8080, function(){ //8080포트에서 통신이 오면 들어라.
//     console.log('listening on 8080');
// });  

// //요청 처리 작성 -> 특정 경로로 방문하면 
// // url을 입력하는 것 -> get요청을 하는 것임!
// // /pet으로 접속하면 안내문을 띄워라
 
// app.get('/pet', 
//     function(req, res){  //요청, 응답
//     //pet이라는 경로에 방문을 하면 안내문을 띄워주는 코드!
//         res.send('반갑습니다(pet page)');
//     }   
// );


//서버를 열 수 있다.
//8080이라는 포트에 서버를 띄우라는 말임.
//60000개 중 8080번 포트로 진입한다.
//크롬에서 http://localhost:8080/을 입력하면 진입 가능! 
