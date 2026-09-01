# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# NOTE: `expression.lldb` hardcodes the line number of the `# breakpoint`
# marker below. Adding or reordering lines requires updating it.

comptime kScale = 10


@fieldwise_init
struct Point(Copyable, Movable):
    var x: Int
    var y: Int

    def sum(self) -> Int:
        return self.x + self.y


def add(lhs: Int, rhs: Int) -> Int:
    return lhs + rhs


def main():
    var lhs = 20
    var rhs = 22
    var count = UInt(7)
    var flag = True
    var ratio = 2.5
    var point = Point(3, 4)
    var numbers = [10, 20, 30]
    var text = String("hello")
    var answer = add(lhs, rhs)
    print(answer, count, flag, ratio, point.x, numbers[0], text)  # breakpoint
