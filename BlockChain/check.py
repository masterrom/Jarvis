from eth_rpc_api import EthJsonRpc

client = EthJsonRpc('127.0.0.1', 8545)
print(client.eth_accounts)



# # Import
# from web3 import Web3, HTTPProvider

# # Connection to the remote server
# web3rpc = Web3(HTTPProvider(host="0.0.0.0", port="8545")) 

# # Unlock your account
# duration = 1000
# web3rpc.personal.unlockAccount(web3rpc.eth.accounts[0], '1234', duration)

# print("connected Success")