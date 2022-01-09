#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define size_message_length_ 1024 // Buffer size for the length [B]
#define size_sz_length_ 4 // Buffer size for the message length content [B]

namespace socket_communication
{

class Client
{
 public:
  Client();
  Client(const std::string ip, int port);
  ~Client();

  void Init(const std::string ip="127.0.0.1",int port=1234);
  void Send(std::string message);
  std::string Receive();

 private:
  int client_;
};

Client::Client() {}

Client::Client(const std::string ip, int port)
{
  Init(ip,port);
}

Client::~Client()
{
  close(client_);
}

void Client::Init(const std::string ip,int port)
{
  client_ = socket(AF_INET, SOCK_STREAM, 0);
  if (client_ < 0)
  {
    std::cout << "[Client]: ERROR establishing socket\n";
    exit(1);
  }

  bool connected = false;
  int max_connection_attempts = 5;

  while ((!connected) && (max_connection_attempts > 0))
  {
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr);

    if (connect(client_, (const struct sockaddr*)&serv_addr,
                sizeof(serv_addr)) == 0)
    {
      connected = true;
      std::cout << "[Client]: Cpp socket client connected." << std::endl;
    }
    else
    {
      port++;
      max_connection_attempts--;
      std::cout << "[Client]: Error connecting to port " << port-1
                << ". Attempting to connet to port: " << port << "\n";
    }
  }
}

void Client::Send(std::string message)
{
  std::string length_str = std::to_string(message.length());
  std::string message_length =
      std::string(size_sz_length_-length_str.length(),'0')+length_str;
  std::string the_message = message_length+message;  
  send(client_, the_message.c_str(), size_message_length_, 0);
  // std::cout << the_message << "\n";
}

std::string Client::Receive()
{
  // Receive message
  char message_raw[size_message_length_] = {0};
  int n = recv(client_, message_raw, size_message_length_, 0);
  std::string message_raw_str = std::string(message_raw);
  // Get length of message
  std::string msg_length = message_raw_str.substr(0,size_sz_length_);
  // for (int i=0;i<size_sz_length_;i++)
  // {
  //   msg_length.append(std::string(message_raw[i]));
  // }
  int length = std::stoi(msg_length);
  std::string msg_content = message_raw_str.substr(
      size_sz_length_,size_sz_length_+length);
  return msg_content;
}

}

void write_to_file(std::string msg)
{
  std::ofstream out("result.tex");
  out << "\\xdef\\out{" << msg << "}%";
  out.close();
}

int main(int argc, char** argv)
{
  // Read command
  std::string cmd = "";
  for (int i=1;i<argc;i++)
  {
    cmd.append(std::string(argv[i]).append(" "));
  }

  // Send command to Python and receive reply
  socket_communication::Client client("127.0.0.1",1234);
  client.Send(cmd);

  // Receive response
  std::string msg = client.Receive();
  std::cout << "[Client]: Received " << msg << std::endl;
  
  // Save output to file
  write_to_file(msg);

  return 0;
}
