// Copyright 2025 QuantClaw Contributors
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <memory>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include "quantclaw/auth/openai_codex_auth.hpp"
#include "quantclaw/providers/openai_codex_provider.hpp"
#include "quantclaw/providers/provider_error.hpp"

#include "test_helpers.hpp"
#include <gtest/gtest.h>

namespace quantclaw {
namespace {

std::shared_ptr<spdlog::logger> make_logger(const std::string& name) {
  auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
  return std::make_shared<spdlog::logger>(name, sink);
}

class FakeBearerTokenSource : public auth::BearerTokenSource {
 public:
  std::string token = "oauth-token";
  int calls = 0;

  std::string ResolveAccessToken() override {
    ++calls;
    return token;
  }
};

}  // namespace

TEST(OpenAICodexProviderTest, ChatCompletionUsesBearerTokenAndParsesOutput) {
  const int port = test::FindFreePort();
  ASSERT_GT(port, 0);

  httplib::Server server;
  std::atomic<bool> saw_request = false;
  server.Post("/codex/responses", [&](const httplib::Request& req,
                                      httplib::Response& res) {
    saw_request = true;
    EXPECT_EQ(req.get_header_value("Authorization"), "Bearer oauth-token");

    auto body = nlohmann::json::parse(req.body);
    EXPECT_EQ(body.value("model", ""), "gpt-5");
    EXPECT_EQ(body.value("instructions", ""), "You are helpful.");
    EXPECT_EQ(body.value("stream", false), true);
    EXPECT_EQ(body.value("store", true), false);
    EXPECT_FALSE(body.contains("temperature"));
    EXPECT_EQ(body.value("max_output_tokens", 0), 8192);
    ASSERT_TRUE(body.contains("input"));
    ASSERT_TRUE(body["input"].is_array());
    ASSERT_FALSE(body["input"].empty());

    res.set_chunked_content_provider(
        "text/event-stream", [](size_t /*offset*/, httplib::DataSink& sink) {
          const char event1[] =
              "data: {\"type\":\"response.output_text.delta\",\"delta\":"
              "\"Hello from Codex\"}\n\n";
          const char event2[] =
              "data: {\"type\":\"response.completed\",\"response\":"
              "{\"status\":\"completed\",\"usage\":{\"input_tokens\":11,"
              "\"output_tokens\":7,\"total_tokens\":18}}}\n\n";
          sink.write(event1, sizeof(event1) - 1);
          sink.write(event2, sizeof(event2) - 1);
          sink.done();
          return true;
        });
  });

  std::thread server_thread([&]() {
    test::ReleaseHeldPort(port);
    server.listen("127.0.0.1", port);
  });
  ASSERT_TRUE(test::WaitForServerReady(port));

  auto logger = make_logger("openai-codex-provider");
  auto token_source = std::make_shared<FakeBearerTokenSource>();
  OpenAICodexProvider provider("http://127.0.0.1:" + std::to_string(port), 30,
                               logger, token_source);

  ChatCompletionRequest request;
  request.model = "gpt-5";
  request.messages.push_back({"system", "You are helpful."});
  request.messages.push_back({"user", "Say hello"});

  const auto response = provider.ChatCompletion(request);
  EXPECT_TRUE(saw_request.load());
  EXPECT_EQ(token_source->calls, 1);
  EXPECT_EQ(response.content, "Hello from Codex");
  EXPECT_EQ(response.usage.prompt_tokens, 11);
  EXPECT_EQ(response.usage.completion_tokens, 7);
  EXPECT_EQ(response.usage.total_tokens, 18);

  server.stop();
  server_thread.join();
}

TEST(OpenAICodexProviderTest, StreamingParsesTextDeltas) {
  const int port = test::FindFreePort();
  ASSERT_GT(port, 0);

  httplib::Server server;
  server.Post("/codex/responses", [&](const httplib::Request& req,
                                      httplib::Response& res) {
    EXPECT_EQ(req.get_header_value("Authorization"), "Bearer oauth-token");
    auto body = nlohmann::json::parse(req.body);
    EXPECT_EQ(body.value("model", ""), "gpt-5");
    EXPECT_EQ(body.value("instructions", ""), "");
    EXPECT_EQ(body.value("stream", false), true);
    EXPECT_EQ(body.value("store", true), false);
    EXPECT_FALSE(body.contains("temperature"));
    EXPECT_EQ(body.value("max_output_tokens", 0), 8192);
    res.set_chunked_content_provider(
        "text/event-stream", [](size_t /*offset*/, httplib::DataSink& sink) {
          const char event1[] =
              "data: {\"type\":\"response.output_text.delta\",\"delta\":"
              "\"Hello \"}\n\n";
          const char event2[] =
              "data: {\"type\":\"response.output_text.delta\",\"delta\":"
              "\"world\"}\n\n";
          const char event3[] =
              "data: {\"type\":\"response.completed\",\"response\":"
              "{\"status\":\"completed\"}}\n\n";
          sink.write(event1, sizeof(event1) - 1);
          sink.write(event2, sizeof(event2) - 1);
          sink.write(event3, sizeof(event3) - 1);
          sink.done();
          return true;
        });
  });

  std::thread server_thread([&]() {
    test::ReleaseHeldPort(port);
    server.listen("127.0.0.1", port);
  });
  ASSERT_TRUE(test::WaitForServerReady(port));

  auto logger = make_logger("openai-codex-provider-stream");
  auto token_source = std::make_shared<FakeBearerTokenSource>();
  OpenAICodexProvider provider("http://127.0.0.1:" + std::to_string(port), 30,
                               logger, token_source);

  ChatCompletionRequest request;
  request.model = "gpt-5";
  request.stream = true;
  request.messages.push_back({"user", "Say hello"});

  std::string text;
  bool saw_end = false;
  provider.ChatCompletionStream(request,
                                [&](const ChatCompletionResponse& chunk) {
                                  text += chunk.content;
                                  if (chunk.is_stream_end) {
                                    saw_end = true;
                                  }
                                });

  EXPECT_EQ(text, "Hello world");
  EXPECT_TRUE(saw_end);

  server.stop();
  server_thread.join();
}

TEST(OpenAICodexProviderTest, ChatCompletionSkipsOrphanToolResults) {
  const int port = test::FindFreePort();
  ASSERT_GT(port, 0);

  httplib::Server server;
  std::atomic<bool> saw_request = false;
  std::atomic<bool> saw_function_call = false;
  std::atomic<bool> saw_matched_output = false;
  std::atomic<bool> saw_orphan_output = false;
  server.Post("/codex/responses", [&](const httplib::Request& req,
                                      httplib::Response& res) {
    saw_request = true;
    EXPECT_EQ(req.get_header_value("Authorization"), "Bearer oauth-token");

    const auto body = nlohmann::json::parse(req.body);
    ASSERT_TRUE(body.contains("input"));
    ASSERT_TRUE(body["input"].is_array());

    for (const auto& item : body["input"]) {
      if (item.value("type", "") == "function_call" &&
          item.value("call_id", "") == "matched-call") {
        saw_function_call = true;
      }
      if (item.value("type", "") == "function_call_output" &&
          item.value("call_id", "") == "matched-call") {
        saw_matched_output = true;
      }
      if (item.value("type", "") == "function_call_output" &&
          item.value("call_id", "") == "orphan-call") {
        saw_orphan_output = true;
      }
    }

    res.set_chunked_content_provider(
        "text/event-stream", [](size_t /*offset*/, httplib::DataSink& sink) {
          const char event1[] =
              "data: {\"type\":\"response.output_text.delta\",\"delta\":"
              "\"ok\"}\n\n";
          const char event2[] =
              "data: {\"type\":\"response.completed\",\"response\":"
              "{\"status\":\"completed\"}}\n\n";
          sink.write(event1, sizeof(event1) - 1);
          sink.write(event2, sizeof(event2) - 1);
          sink.done();
          return true;
        });
  });

  std::thread server_thread([&]() {
    test::ReleaseHeldPort(port);
    server.listen("127.0.0.1", port);
  });
  ASSERT_TRUE(test::WaitForServerReady(port));

  auto logger = make_logger("openai-codex-provider-orphan-tool");
  auto token_source = std::make_shared<FakeBearerTokenSource>();
  OpenAICodexProvider provider("http://127.0.0.1:" + std::to_string(port), 30,
                               logger, token_source);

  ChatCompletionRequest request;
  request.model = "gpt-5";
  request.messages.push_back({"user", "before"});

  Message assistant_tool_call;
  assistant_tool_call.role = "assistant";
  assistant_tool_call.content.push_back(
      ContentBlock::MakeToolUse("matched-call", "read_file",
                                {{"path", "/tmp/a.txt"}}));
  request.messages.push_back(std::move(assistant_tool_call));

  Message matched_tool_result;
  matched_tool_result.role = "user";
  matched_tool_result.content.push_back(
      ContentBlock::MakeToolResult("matched-call", "file contents"));
  request.messages.push_back(std::move(matched_tool_result));

  Message orphan_tool_result;
  orphan_tool_result.role = "user";
  orphan_tool_result.content.push_back(
      ContentBlock::MakeToolResult("orphan-call", "stale output"));
  request.messages.push_back(std::move(orphan_tool_result));

  request.messages.push_back({"user", "after"});

  const auto response = provider.ChatCompletion(request);
  EXPECT_TRUE(saw_request.load());
  EXPECT_TRUE(saw_function_call.load());
  EXPECT_TRUE(saw_matched_output.load());
  EXPECT_FALSE(saw_orphan_output.load());
  EXPECT_EQ(response.content, "ok");

  server.stop();
  server_thread.join();
}

TEST(OpenAICodexProviderTest,
     ChatCompletionPrefersHttpClassificationWhenSseReportsError) {
  const int port = test::FindFreePort();
  ASSERT_GT(port, 0);

  httplib::Server server;
  server.Post("/codex/responses", [&](const httplib::Request& /*req*/,
                                      httplib::Response& res) {
    res.status = 401;
    res.set_chunked_content_provider(
        "text/event-stream", [](size_t /*offset*/, httplib::DataSink& sink) {
          const char event[] =
              "data: {\"type\":\"error\",\"message\":\"unauthorized\"}\n\n";
          sink.write(event, sizeof(event) - 1);
          sink.done();
          return true;
        });
  });

  std::thread server_thread([&]() {
    test::ReleaseHeldPort(port);
    server.listen("127.0.0.1", port);
  });
  ASSERT_TRUE(test::WaitForServerReady(port));

  auto logger = make_logger("openai-codex-provider-http-error");
  auto token_source = std::make_shared<FakeBearerTokenSource>();
  OpenAICodexProvider provider("http://127.0.0.1:" + std::to_string(port), 30,
                               logger, token_source);

  ChatCompletionRequest request;
  request.model = "gpt-5";
  request.messages.push_back({"user", "Say hello"});

  try {
    (void)provider.ChatCompletion(request);
    FAIL() << "Expected ProviderError";
  } catch (const ProviderError& e) {
    EXPECT_EQ(e.Kind(), ProviderErrorKind::kAuthError);
    EXPECT_EQ(e.HttpStatus(), 401);
  }

  server.stop();
  server_thread.join();
}

TEST(OpenAICodexProviderTest,
     StreamingPrefersHttpClassificationWhenSseReportsError) {
  const int port = test::FindFreePort();
  ASSERT_GT(port, 0);

  httplib::Server server;
  server.Post("/codex/responses", [&](const httplib::Request& /*req*/,
                                      httplib::Response& res) {
    res.status = 429;
    res.set_chunked_content_provider(
        "text/event-stream", [](size_t /*offset*/, httplib::DataSink& sink) {
          const char event[] =
              "data: {\"type\":\"error\",\"message\":\"rate limited\"}\n\n";
          sink.write(event, sizeof(event) - 1);
          sink.done();
          return true;
        });
  });

  std::thread server_thread([&]() {
    test::ReleaseHeldPort(port);
    server.listen("127.0.0.1", port);
  });
  ASSERT_TRUE(test::WaitForServerReady(port));

  auto logger = make_logger("openai-codex-provider-stream-http-error");
  auto token_source = std::make_shared<FakeBearerTokenSource>();
  OpenAICodexProvider provider("http://127.0.0.1:" + std::to_string(port), 30,
                               logger, token_source);

  ChatCompletionRequest request;
  request.model = "gpt-5";
  request.stream = true;
  request.messages.push_back({"user", "Say hello"});

  try {
    provider.ChatCompletionStream(
        request, [](const ChatCompletionResponse& /*chunk*/) {});
    FAIL() << "Expected ProviderError";
  } catch (const ProviderError& e) {
    EXPECT_EQ(e.Kind(), ProviderErrorKind::kRateLimit);
    EXPECT_EQ(e.HttpStatus(), 429);
  }

  server.stop();
  server_thread.join();
}

}  // namespace quantclaw
