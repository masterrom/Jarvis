Web3 = require('web3');
web3 = new Web3(new Web3.providers.HttpProvider("http://127.0.0.1:8545"));
// console.log(web3.eth.getAccounts());
var defaultAccount = web3.eth.getAccounts(console.log);

var items = ["shoe", "person", "belt", "gun", "sharp"]
// using the promise
web3.eth.sendTransaction({
    from: defaultAccount,
    to: defaultAccount,
    data: "" + new Date() + " " + items[Math.random]
})
.then(function(receipt){
    console.log("Log has been submitted to blockchain")
});
